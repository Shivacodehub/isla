"""
protect.py  (ISLA — Identity-Sensitive head Localization for multi-subject APDM)
=================================================================================
Drop-in replacement for the original APDM protect.py.
Adds:
  1. Phase 0: ArcFace head attribution + per-head projection matrices (identity_heads.py)
  2. Per-subject identity suppression loss (L_ID) with hinge formulation
  3. Per-head regularization loss (L_head)
  4. MGDA Frank-Wolfe gradient conflict resolution across N=4 subjects (mgda.py)
  5. Modified UNet forward hooks to project selected head outputs during training

New args (all optional, have defaults):
  --num_subjects          int   number of subjects (default 4)
  --subject_data_dirs     list  space-separated paths to each subject's images
  --subject_prompts       list  space-separated quoted prompts per subject
  --arcface_model_path    str   path to ArcFace .pth checkpoint
  --top_k_heads           int   number of identity heads to select (default 16)
  --lambda_id             float weight for L_ID (default 1.0)
  --lambda_head           float weight for L_head (default 0.1)
  --id_tau                float hinge threshold for identity suppression (default 0.3)
  --phase0_timesteps      list  timesteps for head attribution scoring
  --isla_mode             bool  flag to enable ISLA (if not set, falls back to APDM)

All original APDM args are preserved unchanged.
"""

import torch
import random
import os
import itertools
import sys
import numpy as np
import torch.nn.functional as F
from argparse import ArgumentParser
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from PIL import Image
from torchvision.utils import save_image
from torchvision import transforms
from datetime import datetime
from diffusers.schedulers import DDPMScheduler
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
from tqdm import tqdm

# ISLA modules
from identity_heads import run_phase0, project_head_output, head_regularization_loss
from mgda import extract_flat_grad, apply_flat_grad, combine_gradients_mgda


# ===========================================================================
# Datasets (unchanged from APDM)
# ===========================================================================

class ImageDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.images = os.listdir(data_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.data_dir, self.images[idx])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


class PairedImageDataset(Dataset):
    def __init__(self, data_dir1, data_dir2, size=512):
        self.data_dir1 = data_dir1
        self.data_dir2 = data_dir2
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop((size, size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.images1 = os.listdir(data_dir1)
        self.images2 = os.listdir(data_dir2)
        assert len(self.images1) == len(self.images2)

    def __len__(self):
        return len(self.images1)

    def __getitem__(self, idx):
        img1 = Image.open(os.path.join(self.data_dir1, self.images1[idx])).convert("RGB")
        img2 = Image.open(os.path.join(self.data_dir2, self.images2[idx])).convert("RGB")
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2


# ===========================================================================
# Utilities (unchanged from APDM)
# ===========================================================================

def encode_prompt(tokenizer, text_encoder, prompt, do_classifier_free_guidance,
                  num_images_per_prompt=1):
    text_inputs = tokenizer(
        prompt, padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt"
    ).to("cuda")
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to("cuda"), attention_mask=None)[0]
    bs_embeds, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embeds * num_images_per_prompt, seq_len, -1)
    if do_classifier_free_guidance:
        negative_text_inputs = tokenizer(
            "", padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        ).to("cuda")
        negative_prompt_embeds = text_encoder(
            negative_text_inputs.input_ids.to("cuda"), attention_mask=None
        )[0]
        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(
            bs_embeds * num_images_per_prompt, seq_len, -1
        )
    else:
        negative_prompt_embeds = None
    return prompt_embeds, negative_prompt_embeds


def sample_data(loader):
    while True:
        for data in loader:
            yield data


# ===========================================================================
# ArcFace loader  (insightface-native)
# ===========================================================================

class InsightFaceWrapper(torch.nn.Module):
    """
    Wraps insightface's FaceAnalysis so it behaves like our ArcFace interface:
        embeddings = model(face_tensor)   # (B, 512) normalised, on CUDA

    insightface works on numpy uint8 BGR images, so we convert inside forward.
    Input: float32 tensor (B, 3, 112, 112) in [-1, 1] range (RGB).
    Output: float32 tensor (B, 512) L2-normalised, on same device as input.
    """
    def __init__(self, app, device="cuda"):
        super().__init__()
        self.app = app          # insightface FaceAnalysis object
        self.device = device
        # insightface's recognition model is stored in app.models['recognition']
        self.rec_model = None
        for name, model in app.models.items():
            if "recognition" in name.lower() or "arcface" in name.lower():
                self.rec_model = model
                break
        # Fallback: grab first model that has 'get_feat'
        if self.rec_model is None:
            for model in app.models.values():
                if hasattr(model, "get_feat"):
                    self.rec_model = model
                    break
        if self.rec_model is None:
            raise RuntimeError(
                "Could not find recognition model in insightface FaceAnalysis. "
                "Make sure 'buffalo_l' or 'buffalo_sc' pack is downloaded."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, 112, 112) float32 tensor in [-1, 1], RGB
        returns: (B, 512) float32 tensor, L2-normalised
        """
        import numpy as np
        import cv2

        B = x.shape[0]
        embs = []
        x_np = (x.detach().cpu().numpy() * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        # x_np: (B, 3, 112, 112)  ->  convert to (B, 112, 112, 3) BGR for insightface
        for b in range(B):
            img_rgb = x_np[b].transpose(1, 2, 0)          # (112, 112, 3) RGB
            img_bgr = img_rgb[:, :, ::-1].copy()           # BGR
            feat = self.rec_model.get_feat(img_bgr)        # (1, 512) numpy
            embs.append(torch.from_numpy(feat).squeeze(0)) # (512,)

        embs_t = torch.stack(embs, dim=0).float().to(x.device)   # (B, 512)
        return torch.nn.functional.normalize(embs_t, dim=1)


def load_arcface(arcface_model_path=None, device="cuda"):
    """
    Load ArcFace as pure PyTorch iresnet50 — no ONNX runtime, runs on CUDA 12.2.
    Uses arcface_torch.py which auto-converts the cached ONNX file.
    """
    from arcface_torch import load_arcface_torch
    pth_path = arcface_model_path if (arcface_model_path and
                                       arcface_model_path != "none") else None
    return load_arcface_torch(pth_path=pth_path, device=device)


def _dummy_arcface(device="cuda"):
    """Random ResNet for pipeline testing when ArcFace unavailable."""
    import torchvision.models as tvm
    m = tvm.resnet18(pretrained=False)
    m.fc = torch.nn.Linear(512, 512)
    m = m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    print("[WARN] Using dummy ArcFace — identity losses will be random.")
    return m


# ===========================================================================
# UNet hook manager for TRAINING (projects selected heads during forward)
# ===========================================================================

class ISLAHookManager:
    """
    Registers forward hooks on selected attention head modules.
    During each forward pass:
      - captures z_h (the head output before concat)
      - replaces it with z_h* = project_head_output(z_h, Q_h)

    Also keeps a dict of unmodified z_h for L_head computation.

    Works with diffusers BasicTransformerBlock -> attn1 (self-attention).
    The Attention module in diffusers concatenates multi-head outputs internally,
    so we hook into the head-level via the processor approach.

    Implementation: patch the `forward` of each self-attn Attention module to
    intercept and project per-head outputs.
    """

    def __init__(self, unet, selected_heads, Q_h_dict):
        self.unet = unet
        self.selected_heads = set(selected_heads)
        self.Q_h_dict = Q_h_dict
        self.z_h_store = {}        # (l, h) -> z_h tensor WITH grad (for L_head)
        self.hooks = []
        self._layer_counter = [0]  # mutable for closure
        self._register()

    def _make_attn_hook(self, layer_idx, attn_module):
        """
        Hook on the Attention module (attn1 = self-attn).
        Intercepts after the module's forward, re-computes per-head output,
        projects selected heads, and reconstructs the concatenated output.
        """
        def hook(module, input, output):
            hidden = input[0]                          # (B, N_tok, d_model)
            B, N_tok, d_model = hidden.shape
            num_heads = module.heads
            d_k = d_model // num_heads

            # Re-derive per-head outputs (differentiable)
            q = module.to_q(hidden)
            k = module.to_k(hidden)
            v = module.to_v(hidden)

            def split(x):
                x = x.view(B, N_tok, num_heads, d_k)
                return x.permute(0, 2, 1, 3).reshape(B * num_heads, N_tok, d_k)

            q_h = split(q)
            k_h = split(k)
            v_h = split(v)

            scale = d_k ** -0.5
            attn_w = torch.softmax(
                torch.bmm(q_h, k_h.transpose(-1, -2)) * scale, dim=-1
            )
            z_all = torch.bmm(attn_w, v_h)              # (B*H, N_tok, d_k)
            z_all = z_all.view(B, num_heads, N_tok, d_k) # (B, H, N_tok, d_k)

            # Project selected heads
            z_parts = []
            for h in range(num_heads):
                z_h = z_all[:, h, :, :]                  # (B, N_tok, d_k)
                key = (layer_idx, h)

                # Store for L_head
                self.z_h_store[key] = z_h

                if key in self.selected_heads and self.Q_h_dict.get(key) is not None:
                    Q_h = self.Q_h_dict[key]             # (N_tok*d_k, N)
                    z_h = project_head_output(z_h, Q_h)

                z_parts.append(z_h)

            # Reconstruct: concat heads -> (B, N_tok, d_model)
            z_cat = torch.cat(z_parts, dim=-1)           # (B, N_tok, H*d_k)

            # Apply output projection (to_out in diffusers Attention)
            out = module.to_out[0](z_cat)
            if len(module.to_out) > 1:
                out = module.to_out[1](out)               # dropout (identity at eval)

            return out

        return hook

    def _register(self):
        idx = 0
        for name, module in self.unet.named_modules():
            if "attn1" in name and hasattr(module, "to_q") and hasattr(module, "heads"):
                h = module.register_forward_hook(self._make_attn_hook(idx, module))
                self.hooks.append(h)
                idx += 1

    def clear_store(self):
        self.z_h_store.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ===========================================================================
# Identity suppression: fast image generation for L_ID
# ===========================================================================

@torch.no_grad()
def fast_decode_identity(unet_temp, vae, ddpm_scheduler, arcface,
                          z_t, t, prompt_emb):
    """
    One-step denoising approximation -> decode -> ArcFace embedding.
    Used for L_ID computation. No grad needed through generation path
    (we supervise through hook on z_h which has grad).
    """
    alpha_bar = ddpm_scheduler.alphas_cumprod[t.item()].to(z_t.device)
    with torch.no_grad():
        eps_pred = unet_temp(z_t, t, encoder_hidden_states=prompt_emb,
                             return_dict=False)[0]
    z0_pred = (z_t - (1 - alpha_bar).sqrt() * eps_pred) / alpha_bar.sqrt()
    z0_pred = z0_pred.clamp(-4, 4)
    img = vae.decode(z0_pred / vae.config.scaling_factor).sample
    img = (img * 0.5 + 0.5).clamp(0, 1)
    img_face = F.interpolate(img, size=112, mode="bilinear", align_corners=False)
    img_face_norm = (img_face - 0.5) / 0.5
    e_gen = arcface(img_face_norm)
    e_gen = F.normalize(e_gen, dim=1).squeeze(0)
    return e_gen


# ===========================================================================
# Argument parsing
# ===========================================================================

def parse_args():
    parser = ArgumentParser()
    # ── Original APDM args ──────────────────────────────────────────────────
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="models/stable-diffusion-v1-5")
    parser.add_argument("--instance_data_dir", type=str, default="data/person")
    parser.add_argument("--instance_prompt", type=str,
                        default="a photo of sks person")
    parser.add_argument("--with_prior_preservation", action="store_true")
    parser.add_argument("--prior_loss_weight", type=float, default=1.0)
    parser.add_argument("--class_data_dir", type=str,
                        default="class_images/person")
    parser.add_argument("--exp", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--iter", type=int, default=801)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_text_encoder", action="store_true")
    parser.add_argument("--num_inner_iter", type=int, default=30)
    parser.add_argument("--negative_loss", action="store_true")
    parser.add_argument("--print_freq", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=100000)
    parser.add_argument("--relu_bound", type=float, default=None)
    parser.add_argument("--class_prompt", type=str,
                        default="a photo of a person")
    parser.add_argument("--grad_accum_type", type=str, default="sum")
    parser.add_argument("--unfreeze", nargs="+", default=None)
    parser.add_argument("--in_ppl", action="store_true")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--loss_dpo", action="store_true")
    parser.add_argument("--loss_dpo_paired_dataset", action="store_true")
    parser.add_argument("--loss_dpo_paired_dataset_dir", type=str,
                        default="paired_class_images/dog")
    parser.add_argument("--loss_dpo_beta", type=float, default=100)

    # ── New ISLA args ────────────────────────────────────────────────────────
    parser.add_argument("--isla_mode", action="store_true",
                        help="Enable ISLA multi-subject protection")
    parser.add_argument("--num_subjects", type=int, default=4)
    parser.add_argument("--subject_data_dirs", nargs="+",
                        default=None,
                        help="Paths to each subject's image dir (N paths)")
    parser.add_argument("--subject_prompts", nargs="+",
                        default=None,
                        help="Prompt string per subject (N prompts, quoted)")
    parser.add_argument("--arcface_model_path", type=str, default=None,
                        help="Path to ArcFace .pth checkpoint")
    parser.add_argument("--top_k_heads", type=int, default=16,
                        help="Number of identity-sensitive heads to select")
    parser.add_argument("--lambda_id", type=float, default=1.0,
                        help="Weight for L_ID (identity suppression)")
    parser.add_argument("--lambda_head", type=float, default=0.1,
                        help="Weight for L_head (head regularization)")
    parser.add_argument("--lambda_pres", type=float, default=0.5,
                        help="Weight for L_pres (preservation loss)")
    parser.add_argument("--id_tau", type=float, default=0.3,
                        help="Hinge threshold for identity suppression loss")
    parser.add_argument("--phase0_timesteps", nargs="+", type=int,
                        default=[100, 300, 500, 700, 900])

    args = parser.parse_args()

    # Auto-fill subject_data_dirs / subject_prompts if ISLA but not provided
    if args.isla_mode:
        if args.subject_data_dirs is None:
            # Fall back to repeating instance_data_dir for all subjects
            args.subject_data_dirs = [args.instance_data_dir] * args.num_subjects
            print("[WARN] --subject_data_dirs not set; using instance_data_dir for all subjects.")
        if args.subject_prompts is None:
            args.subject_prompts = [args.instance_prompt] * args.num_subjects
            print("[WARN] --subject_prompts not set; using instance_prompt for all subjects.")

    return args


# ===========================================================================
# Main
# ===========================================================================

def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── Class image generation (unchanged) ──────────────────────────────────
    if not os.path.exists(args.class_data_dir):
        os.makedirs(args.class_data_dir)
    cur_class_images = len(list(os.listdir(args.class_data_dir)))
    if cur_class_images < args.num_samples:
        pipe_gen = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path, safety_checker=None,
            torch_dtype=torch.float16
        ).to("cuda")
        pipe_gen.set_progress_bar_config(disable=True)
        num_new_images = args.num_samples - cur_class_images
        print(f"Generating {num_new_images} class images...")
        with torch.no_grad():
            for num in tqdm(range(num_new_images)):
                imgs = pipe_gen(prompt=args.class_prompt,
                                num_inference_steps=50).images[0]
                imgs.save(os.path.join(args.class_data_dir,
                                       f"{cur_class_images + num}.png"))
        del pipe_gen
        torch.cuda.empty_cache()

    # ── Load pipeline ────────────────────────────────────────────────────────
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path, safety_checker=None,
        torch_dtype=torch.float16
    ).to("cuda")
    pipe.safety_checker = None
    # Cast trainable components to float32 — fp16 was only for fast loading
    pipe.unet = pipe.unet.float()
    pipe.text_encoder = pipe.text_encoder.float()
    pipe.vae = pipe.vae.float()

    # ── Data loaders ─────────────────────────────────────────────────────────
    image_dataset = ImageDataset(args.instance_data_dir)
    image_loader = DataLoader(image_dataset, batch_size=args.batch_size,
                              shuffle=True)
    image_loader = sample_data(image_loader)

    if args.with_prior_preservation or args.in_ppl or args.isla_mode:
        class_image_dataset = ImageDataset(args.class_data_dir)
        class_image_loader = DataLoader(class_image_dataset,
                                        batch_size=args.batch_size, shuffle=True)
        class_image_loader = sample_data(class_image_loader)

    if args.loss_dpo and args.loss_dpo_paired_dataset:
        paired_image_dataset = PairedImageDataset(
            args.instance_data_dir, args.loss_dpo_paired_dataset_dir
        )
        paired_image_loader = DataLoader(paired_image_dataset,
                                         batch_size=args.batch_size, shuffle=True)
        paired_image_loader = sample_data(paired_image_loader)

    # ── Save path ─────────────────────────────────────────────────────────────
    if args.exp is None:
        save_path = f"experiments/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        save_path = f"experiments/{args.exp}"
    os.makedirs(save_path, exist_ok=True)
    with open(f"{save_path}/args.txt", "w") as f:
        f.write(str(args))

    # ── Scheduler & optimizer ─────────────────────────────────────────────────
    ddpm_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    optimizer = (
        torch.optim.AdamW(
            itertools.chain(pipe.unet.parameters(),
                            pipe.text_encoder.parameters()),
            lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8
        )
        if args.train_text_encoder
        else torch.optim.AdamW(
            pipe.unet.parameters(),
            lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8
        )
    )

    if args.loss_dpo:
        unet_source = deepcopy(pipe.unet)
        unet_source.requires_grad_(False)
        unet_source.eval()

    vae = pipe.vae
    for param in vae.parameters():
        param.requires_grad = False

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    for param in text_encoder.parameters():
        param.requires_grad = False

    # ── ISLA Phase 0 ─────────────────────────────────────────────────────────
    head_info = None
    isla_hooks = None
    arcface = None

    if args.isla_mode:
        print("\n" + "="*60)
        print("ISLA MODE: Running Phase 0 (head attribution)")
        print("="*60)

        arcface = load_arcface(args.arcface_model_path)

        head_info = run_phase0(
            pipe=pipe,
            arcface=arcface,
            subject_dirs=args.subject_data_dirs,
            subject_prompts=args.subject_prompts,
            device="cuda",
            K=args.top_k_heads,
            sample_timesteps=tuple(args.phase0_timesteps),
        )

        # Per-subject image loaders for ISLA inner loop
        subject_loaders = []
        for sd in args.subject_data_dirs:
            ds = ImageDataset(sd)
            dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
            subject_loaders.append(sample_data(dl))

        # Encode all subject prompts once
        subject_prompt_embs = []
        for prompt in args.subject_prompts:
            emb, _ = encode_prompt(tokenizer, text_encoder, prompt, False)
            subject_prompt_embs.append(emb.detach())

        # Frozen reference UNet for L_pres — keep on CPU to save GPU memory
        unet_ref = deepcopy(pipe.unet).cpu()
        unet_ref.requires_grad_(False)
        unet_ref.eval()

        print("[Phase 0] Complete. Starting protection training...\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    start_time = datetime.now()

    for i in range(args.iter):
        optimizer.zero_grad()

        unet_temp = deepcopy(pipe.unet)
        unet_temp.enable_gradient_checkpointing()   # halves activation memory
        optimizer_temp = torch.optim.AdamW(
            unet_temp.parameters(), lr=args.lr,
            betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8
        )

        # ── Register ISLA hooks on unet_temp ────────────────────────────────
        if args.isla_mode and head_info is not None:
            isla_hooks = ISLAHookManager(
                unet_temp,
                head_info["selected_heads"],
                head_info["Q_h_dict"],
            )

        # ── Inner loop: DB fine-tune on unet_temp ───────────────────────────
        for j in range(args.num_inner_iter):
            optimizer_temp.zero_grad()

            data = next(image_loader).to("cuda")
            t = torch.randint(
                0, ddpm_scheduler.config.num_train_timesteps,
                (args.batch_size,), device="cuda"
            ).long()
            z_0 = vae.encode(data).latent_dist.sample() * vae.config.scaling_factor
            eps = torch.randn_like(z_0)
            z_t = ddpm_scheduler.add_noise(z_0, eps, t)

            prompt_embeds, _ = encode_prompt(tokenizer, text_encoder,
                                             args.instance_prompt, False)
            model_pred = unet_temp(z_t, t, encoder_hidden_states=prompt_embeds,
                                   return_dict=False)[0]

            target = eps if ddpm_scheduler.config.prediction_type == "epsilon" \
                else ddpm_scheduler.get_velocity(model_pred, eps, t)

            loss_db = F.mse_loss(model_pred, target, reduction="mean")

            if args.with_prior_preservation:
                class_data = next(class_image_loader).to("cuda")
                z_0_cls = vae.encode(class_data).latent_dist.sample() * vae.config.scaling_factor
                eps_cls = torch.randn_like(z_0_cls)
                z_t_cls = ddpm_scheduler.add_noise(z_0_cls, eps_cls, t)
                prompt_embeds_cls, _ = encode_prompt(tokenizer, text_encoder,
                                                     args.class_prompt, False)
                model_pred_cls = unet_temp(z_t_cls, t,
                                           encoder_hidden_states=prompt_embeds_cls,
                                           return_dict=False)[0]
                target_cls = eps_cls if ddpm_scheduler.config.prediction_type == "epsilon" \
                    else ddpm_scheduler.get_velocity(model_pred_cls, eps_cls, t)
                loss_cls = F.mse_loss(model_pred_cls, target_cls, reduction="mean")
                loss_db = loss_db + args.prior_loss_weight * loss_cls

            loss_db.backward()
            optimizer_temp.step()

        # ── Outer: compute negative / protection loss ─────────────────────────
        data = next(image_loader).to("cuda")
        t = torch.randint(
            0, ddpm_scheduler.config.num_train_timesteps,
            (args.batch_size,), device="cuda"
        ).long()
        z_0 = vae.encode(data).latent_dist.sample() * vae.config.scaling_factor
        eps = torch.randn_like(z_0)
        z_t = ddpm_scheduler.add_noise(z_0, eps, t)
        prompt_embeds, _ = encode_prompt(tokenizer, text_encoder,
                                         args.instance_prompt, False)
        model_pred = unet_temp(z_t, t, encoder_hidden_states=prompt_embeds,
                               return_dict=False)[0]
        target = eps if ddpm_scheduler.config.prediction_type == "epsilon" \
            else ddpm_scheduler.get_velocity(model_pred, eps, t)

        loss_neg = torch.tensor(0.0, device="cuda")

        # Original APDM negative loss
        if args.negative_loss:
            if args.relu_bound is None or args.relu_bound == 0:
                loss_neg = loss_neg - F.mse_loss(model_pred, target, reduction="mean")
            else:
                loss_neg = loss_neg + F.relu(
                    args.relu_bound - F.mse_loss(model_pred, target, reduction="mean")
                )

        if args.in_ppl:
            class_data = next(class_image_loader).to("cuda")
            z_0_cls = vae.encode(class_data).latent_dist.sample() * vae.config.scaling_factor
            eps_cls = torch.randn_like(z_0_cls)
            z_t_cls = ddpm_scheduler.add_noise(z_0_cls, eps_cls, t)
            prompt_embeds_cls, _ = encode_prompt(tokenizer, text_encoder,
                                                 args.class_prompt, False)
            model_pred_cls = unet_temp(z_t_cls, t,
                                       encoder_hidden_states=prompt_embeds_cls,
                                       return_dict=False)[0]
            target_cls = eps_cls if ddpm_scheduler.config.prediction_type == "epsilon" \
                else ddpm_scheduler.get_velocity(model_pred_cls, eps_cls, t)
            loss_cls = F.mse_loss(model_pred_cls, target_cls, reduction="mean")
            loss_neg = loss_neg + args.prior_loss_weight * loss_cls

        # DPO loss (unchanged from APDM)
        if args.loss_dpo:
            if args.loss_dpo_paired_dataset:
                unsafe_data, safe_data = next(paired_image_loader)
                unsafe_data = unsafe_data.to("cuda")
                safe_data = safe_data.to("cuda")
            else:
                unsafe_data = next(image_loader).to("cuda")
                safe_data = next(class_image_loader).to("cuda")

            with torch.no_grad():
                z_0_unsafe = vae.encode(unsafe_data).latent_dist.sample() * vae.config.scaling_factor
                z_0_safe = vae.encode(safe_data).latent_dist.sample() * vae.config.scaling_factor
                eps_dpo = torch.randn_like(z_0_unsafe)
                t_dpo = torch.randint(
                    0, ddpm_scheduler.config.num_train_timesteps,
                    (args.batch_size,), device="cuda"
                ).long()
                z_t_unsafe = ddpm_scheduler.add_noise(z_0_unsafe, eps_dpo, t_dpo)
                z_t_safe = ddpm_scheduler.add_noise(z_0_safe, eps_dpo, t_dpo)
                model_pred_unsafe_source = unet_source(
                    z_t_unsafe, t_dpo, encoder_hidden_states=prompt_embeds,
                    return_dict=False
                )[0]
                model_pred_safe_source = unet_source(
                    z_t_safe, t_dpo, encoder_hidden_states=prompt_embeds,
                    return_dict=False
                )[0]

            model_pred_unsafe_target = unet_temp(
                z_t_unsafe, t_dpo, encoder_hidden_states=prompt_embeds,
                return_dict=False
            )[0]
            model_pred_safe_target = unet_temp(
                z_t_safe, t_dpo, encoder_hidden_states=prompt_embeds,
                return_dict=False
            )[0]

            loss_dpo = (
                F.mse_loss(eps_dpo, model_pred_safe_target, reduction="none")
                - F.mse_loss(eps_dpo, model_pred_safe_source, reduction="none")
                - F.mse_loss(eps_dpo, model_pred_unsafe_target, reduction="none")
                + F.mse_loss(eps_dpo, model_pred_unsafe_source, reduction="none")
            )
            loss_dpo = -F.logsigmoid(-args.loss_dpo_beta * loss_dpo).mean()
            loss_neg = loss_neg + loss_dpo

        # ── ISLA losses ───────────────────────────────────────────────────────
        if args.isla_mode and head_info is not None and arcface is not None:

            flat_grads_subjects = []

            for subj_idx in range(args.num_subjects):
                unet_temp.zero_grad()
                if isla_hooks:
                    isla_hooks.clear_store()

                # Subject data
                s_data = next(subject_loaders[subj_idx]).to("cuda")
                t_s = torch.randint(
                    0, ddpm_scheduler.config.num_train_timesteps,
                    (args.batch_size,), device="cuda"
                ).long()
                with torch.no_grad():
                    z_0_s = vae.encode(s_data).latent_dist.sample() \
                            * vae.config.scaling_factor
                eps_s = torch.randn_like(z_0_s)
                z_t_s = ddpm_scheduler.add_noise(z_0_s, eps_s, t_s)

                p_emb = subject_prompt_embs[subj_idx]

                # Forward through modified UNet
                model_pred_s = unet_temp(
                    z_t_s, t_s, encoder_hidden_states=p_emb,
                    return_dict=False
                )[0]

                # L_DB per subject (keep model functional)
                loss_db_s = F.mse_loss(model_pred_s, eps_s)

                # L_ID: identity suppression
                e_gen = fast_decode_identity(
                    unet_temp, vae, ddpm_scheduler, arcface,
                    z_t_s, t_s, p_emb
                )
                mean_emb = head_info["mean_embeddings"][subj_idx]
                sim = torch.dot(F.normalize(e_gen, dim=0), mean_emb.detach())
                loss_id_s = torch.clamp(sim - args.id_tau, min=0.0)

                # L_head: head regularization
                loss_head_s = head_regularization_loss(
                    isla_hooks.z_h_store,
                    head_info["Q_h_dict"],
                    head_info["selected_heads"]
                ) if isla_hooks else torch.tensor(0.0, device="cuda")

                # L_pres: preservation vs frozen reference (unet_ref on CPU)
                class_data_pres = next(class_image_loader).to("cuda")
                with torch.no_grad():
                    z_0_pres = vae.encode(class_data_pres).latent_dist.sample() \
                               * vae.config.scaling_factor
                eps_pres = torch.randn_like(z_0_pres)
                z_t_pres = ddpm_scheduler.add_noise(z_0_pres, eps_pres, t_s)
                prompt_embs_cls, _ = encode_prompt(
                    tokenizer, text_encoder, args.class_prompt, False
                )
                # unet_ref on CPU — move inputs to CPU, run, move result back
                with torch.no_grad():
                    feats_ref = unet_ref(
                        z_t_pres.cpu(), t_s.cpu(),
                        encoder_hidden_states=prompt_embs_cls.cpu(),
                        return_dict=False
                    )[0].to("cuda")
                feats_new = unet_temp(
                    z_t_pres, t_s,
                    encoder_hidden_states=prompt_embs_cls,
                    return_dict=False
                )[0]
                loss_pres_s = F.mse_loss(feats_new, feats_ref.detach())

                loss_subj = (
                    loss_db_s
                    + args.lambda_id * loss_id_s
                    + args.lambda_head * loss_head_s
                    + args.lambda_pres * loss_pres_s
                )
                loss_subj.backward()

                # Extract grad to CPU immediately to free GPU memory
                flat_g = extract_flat_grad(unet_temp).clone()  # CPU tensor
                flat_grads_subjects.append(flat_g)
                unet_temp.zero_grad()
                torch.cuda.empty_cache()

            # MGDA on CPU
            combined_grad, alpha = combine_gradients_mgda(
                flat_grads_subjects, unet_temp, device="cuda"
            )

            # Add negative loss gradient
            if loss_neg.requires_grad:
                loss_neg.backward()
                neg_flat = extract_flat_grad(unet_temp).cpu()
                unet_temp.zero_grad()
                combined_grad = combined_grad + neg_flat

            # Write combined grad directly into pipe.unet (skip unet_temp transfer)
            offset = 0
            for p_main, p_temp in zip(pipe.unet.parameters(),
                                       unet_temp.parameters()):
                numel = p_main.numel()
                chunk = combined_grad[offset: offset + numel].to("cuda").view(p_main.shape)
                if args.unfreeze is None:
                    p_main.grad = chunk.to(p_main.dtype)
                offset += numel

        else:
            loss_neg.backward()
            # Transfer grads unet_temp -> pipe.unet
            with torch.no_grad():
                for (name1, param1), (_, param2) in zip(
                    pipe.unet.named_parameters(), unet_temp.named_parameters()
                ):
                    if param2.grad is not None:
                        if args.unfreeze is None:
                            param1.grad = param2.grad.clone()
                        elif any(name in name1 for name in args.unfreeze):
                            param1.grad = param2.grad.clone()

        if args.grad_accum_type == "mean":
            with torch.no_grad():
                for param in pipe.unet.parameters():
                    if param.grad is not None:
                        param.grad /= args.num_inner_iter

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), max_norm=1.0)
        optimizer.step()

        # ── Logging ───────────────────────────────────────────────────────────
        if i % args.print_freq == 0:
            elapsed = datetime.now() - start_time
            eta = elapsed / (i + 1) * (args.iter - i - 1) if i > 0 else "?"
            print(
                f"Iter {str(i).zfill(3)}: "
                f"Loss(neg)={loss_neg.item():.5f}  "
                f"Time/it={elapsed/(i+1)}  ETA={eta}"
            )
            sys.stdout.flush()

        # ── Save checkpoint ───────────────────────────────────────────────────
        if (i != 0 and i % args.save_freq == 0) or i == args.iter - 1:
            torch.save(pipe.unet.state_dict(),
                       f"{save_path}/unet_{str(i).zfill(3)}.pt")
            if args.train_text_encoder:
                torch.save(pipe.text_encoder.state_dict(),
                           f"{save_path}/text_encoder_{str(i).zfill(3)}.pt")
            if args.isla_mode and head_info is not None:
                # Save head info for reproducibility
                import pickle
                info_save = {
                    "selected_heads": head_info["selected_heads"],
                    "score_matrix": head_info["score_matrix"],
                }
                with open(f"{save_path}/head_info_{str(i).zfill(3)}.pkl", "wb") as f:
                    pickle.dump(info_save, f)

        # Cleanup
        if isla_hooks is not None:
            isla_hooks.remove()
            isla_hooks = None

        del unet_temp
        torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args()
    main(args)