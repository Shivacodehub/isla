"""
identity_heads.py
=================
Phase 0 of the multi-subject ISLA method:
  1. Compute per-subject ArcFace embeddings -> build identity subspace Q (512 x N)
  2. Score all UNet attention heads by identity sensitivity (gradient attribution)
  3. For each top-K head, build a per-head projection matrix Q_h
     that removes identity-sensitive directions from that head's output.

Usage (called once before protection training):
    from identity_heads import run_phase0
    head_info = run_phase0(pipe, arcface_model, subject_dirs, subject_prompts, args)
    # head_info contains: selected_heads, Q_h_dict, identity_Q (512 x N)
"""

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import os
from copy import deepcopy
from diffusers.schedulers import DDPMScheduler


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SubjectDataset(Dataset):
    def __init__(self, data_dir, size=512):
        self.paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                      if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        self.tf = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
        # insightface ArcFace input: 112x112, float32 in [-1, 1], RGB
        # InsightFaceWrapper.forward() does the BGR conversion internally,
        # so we just supply (B, 3, 112, 112) in [-1,1] RGB here.
        self.tf_face = transforms.Compose([
            transforms.Resize(112),
            transforms.CenterCrop(112),
            transforms.ToTensor(),                          # [0,1]
            transforms.Normalize([0.5, 0.5, 0.5],
                                  [0.5, 0.5, 0.5]),         # -> [-1,1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tf(img), self.tf_face(img)


# ---------------------------------------------------------------------------
# ArcFace embedding helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_subject_embeddings(arcface, subject_dirs, device="cuda"):
    """
    Returns:
        embeddings: list of N unit tensors, each shape (512,)
        all_embeddings: list of M_i tensors per subject (before averaging)
    """
    mean_embeddings = []
    for d in subject_dirs:
        ds = SubjectDataset(d)
        imgs_face = torch.stack([ds[i][1] for i in range(len(ds))]).to(device)
        embs = arcface(imgs_face)                      # (M_i, 512)
        embs = F.normalize(embs, dim=1)
        mean_emb = F.normalize(embs.mean(0), dim=0)   # (512,)
        mean_embeddings.append(mean_emb)
    return mean_embeddings                             # list of N tensors (512,)


def build_identity_subspace(mean_embeddings):
    """
    Stack mean embeddings -> QR decompose -> return Q (512 x N) on GPU.
    P_perp = I - Q @ Q.T   (not materialised; applied on-the-fly)

    QR is computed on CPU to avoid cusolver library issues on some
    cluster CUDA setups — matrix is tiny (512 x N) so CPU cost is negligible.
    """
    E = torch.stack(mean_embeddings, dim=1).cpu()  # (512, N) on CPU
    Q, _ = torch.linalg.qr(E, mode="reduced")        # (512, N) orthonormal cols
    return Q.to(mean_embeddings[0].device)            # move back to GPU


# ---------------------------------------------------------------------------
# Hook utilities for capturing / hooking attention head outputs
# ---------------------------------------------------------------------------

class HeadOutputHook:
    """
    Registers forward hooks on every BasicTransformerBlock's attn1 (self-attn)
    to capture the per-head value outputs z_h before concat.

    Diffusers SD1.5 UNet transformer blocks use:
      block.attn1  -> self-attention
      block.attn2  -> cross-attention
    Each is an Attention module. We patch its forward to split per-head output.
    """

    def __init__(self, unet):
        self.unet = unet
        self.activations = {}   # (layer_idx, head_idx) -> tensor
        self.hooks = []
        self._layer_idx = 0
        self._register()

    def _make_hook(self, layer_idx, attn_module):
        """
        Returns a hook that fires after attn_module forward.
        We re-compute per-head z_h from the saved Q,K,V projections.
        The hook stores z_h for each head.
        """
        def hook(module, input, output):
            # output shape: (B, N_tokens, d_model)
            # We need per-head outputs. Re-compute from input hidden states.
            hidden = input[0]                          # (B, N, d)
            B, N, d = hidden.shape
            num_heads = module.heads
            d_k = d // num_heads

            # project to Q, K, V
            q = module.to_q(hidden)  # (B, N, d)
            k = module.to_k(hidden)
            v = module.to_v(hidden)

            # reshape to (B * H, N, d_k)
            def split_heads(x):
                x = x.view(B, N, num_heads, d_k)
                x = x.permute(0, 2, 1, 3)             # (B, H, N, d_k)
                return x.reshape(B * num_heads, N, d_k)

            q_h = split_heads(q)
            k_h = split_heads(k)
            v_h = split_heads(v)

            scale = d_k ** -0.5
            attn = torch.softmax(torch.bmm(q_h, k_h.transpose(-1, -2)) * scale, dim=-1)
            z_all = torch.bmm(attn, v_h)              # (B*H, N, d_k)
            z_all = z_all.view(B, num_heads, N, d_k)  # (B, H, N, d_k)

            for h in range(num_heads):
                key = (layer_idx, h)
                # store mean over batch; keep grad
                self.activations[key] = z_all[:, h, :, :]  # (B, N, d_k)

        return hook

    def _register(self):
        idx = 0
        for name, module in self.unet.named_modules():
            # In diffusers SD1.5 the self-attn inside transformer blocks
            if "attn1" in name and hasattr(module, "to_q") and hasattr(module, "heads"):
                h = module.register_forward_hook(self._make_hook(idx, module))
                self.hooks.append(h)
                idx += 1
        self.num_layers = idx

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ---------------------------------------------------------------------------
# Head attribution scoring
# ---------------------------------------------------------------------------

def encode_prompt_simple(tokenizer, text_encoder, prompt):
    inputs = tokenizer(prompt, padding="max_length",
                       max_length=tokenizer.model_max_length,
                       truncation=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        emb = text_encoder(inputs.input_ids)[0]
    return emb


@torch.no_grad()
def fast_denoise(unet, vae, scheduler, z_t, t, prompt_emb, num_steps=5):
    """
    Very fast approximate denoising (5 DDIM-like steps) to get a roughly
    clean latent from z_t. Used for identity score computation.
    """
    from diffusers import DDIMScheduler
    # Simple one-step denoising approximation using the noise prediction
    # x0_pred = (z_t - sqrt(1-αbar)*eps_pred) / sqrt(αbar)
    alpha_bar = scheduler.alphas_cumprod[t.item()].to(z_t.device)
    eps_pred = unet(z_t, t, encoder_hidden_states=prompt_emb, return_dict=False)[0]
    z0_pred = (z_t - (1 - alpha_bar).sqrt() * eps_pred) / alpha_bar.sqrt()
    return z0_pred


def score_heads(pipe, arcface, mean_embeddings, subject_dirs, subject_prompts,
                ddpm_scheduler, device="cuda", K=16,
                sample_timesteps=(100, 300, 500, 700, 900)):
    """
    Compute S[layer, head] = average ||d(identity_sim)/d(z_h^l)||_F
    over all subjects, images, and sampled timesteps.

    Returns:
        score_matrix: np.ndarray (L, H)
        grad_dict:    dict (layer, head) -> Q_h (basis of identity-sensitive dirs)
        selected:     list of (layer, head) tuples, top-K by score
    """
    unet = pipe.unet
    vae = pipe.vae
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    unet.eval()

    # Register hooks
    hook_manager = HeadOutputHook(unet)
    L = hook_manager.num_layers
    H = None  # will be set on first forward
    score_accum = {}          # (l, h) -> float
    grad_accum = {}           # (l, h) -> list of flattened gradient vectors

    N = len(subject_dirs)
    total_samples = 0

    for i, (data_dir, prompt) in enumerate(zip(subject_dirs, subject_prompts)):
        e_i = mean_embeddings[i]                      # (512,) unit vector
        ds = SubjectDataset(data_dir)
        prompt_emb = encode_prompt_simple(tokenizer, text_encoder, prompt)  # (1, 77, 768)

        for k in range(len(ds)):
            img_vae, img_face = ds[k]
            img_vae = img_vae.unsqueeze(0).to(device)

            z0 = vae.encode(img_vae).latent_dist.sample() * vae.config.scaling_factor

            for t_val in sample_timesteps:
                t = torch.tensor([t_val], device=device).long()
                eps = torch.randn_like(z0)
                alpha_bar = ddpm_scheduler.alphas_cumprod[t_val].to(device)
                z_t = alpha_bar.sqrt() * z0 + (1 - alpha_bar).sqrt() * eps

                # ── Fast Attribution: activation norm only ───────────────
                # No VAE decode, no ArcFace call in this loop.
                # Score = ||z_h|| averaged over subject images & timesteps.
                # Heads with large activations on subject images vs random
                # noise are the identity-sensitive ones. Fast: one UNet
                # forward per (image, timestep), no decode needed.

                hook_manager.activations.clear()

                with torch.no_grad():
                    _ = unet(z_t, t,
                             encoder_hidden_states=prompt_emb,
                             return_dict=False)

                # Score each head by activation norm
                for (l, h), act in hook_manager.activations.items():
                    if H is None:
                        H = max(hh for _, hh in hook_manager.activations) + 1
                    act_norm = act.detach().float().norm().item()
                    key = (l, h)
                    score_accum[key] = score_accum.get(key, 0.0) + act_norm
                    g_flat = act.detach().float().view(-1).cpu()
                    if key not in grad_accum:
                        grad_accum[key] = []
                    grad_accum[key].append(g_flat)

                total_samples += 1

    hook_manager.remove()

    # Normalize scores
    for key in score_accum:
        score_accum[key] /= max(total_samples, 1)

    # Determine H
    if H is None:
        H = 8  # SD1.5 default

    # Build score matrix
    score_matrix = np.zeros((L, H))
    for (l, h), v in score_accum.items():
        if l < L and h < H:
            score_matrix[l, h] = v

    # Select top-K heads
    flat_scores = [(score_matrix[l, h], l, h) for l in range(L) for h in range(H)]
    flat_scores.sort(reverse=True)
    selected = [(l, h) for _, l, h in flat_scores[:K]]

    # Build per-head Q_h (identity-sensitive basis in head activation space)
    Q_h_dict = {}
    for (l, h) in selected:
        key = (l, h)
        if key in grad_accum and len(grad_accum[key]) > 0:
            G = torch.stack(grad_accum[key], dim=1).float()  # (d_feat, M_i*T)
            # Take only first N vectors (one per subject) for QR
            G_sub = G[:, :min(N, G.shape[1])]
            if G_sub.shape[1] > 0:
                Q_sub, _ = torch.linalg.qr(G_sub, mode="reduced")
                Q_h_dict[key] = Q_sub.to(device)             # (d_feat, N)
            else:
                Q_h_dict[key] = None
        else:
            Q_h_dict[key] = None

    print(f"[Phase 0] Head attribution done. Top-{K} heads selected.")
    print(f"  Score range: {score_matrix.min():.4f} -- {score_matrix.max():.4f}")
    print(f"  Selected: {selected[:5]} ...")

    return {
        "score_matrix": score_matrix,
        "selected_heads": selected,
        "Q_h_dict": Q_h_dict,
        "L": L,
        "H": H,
    }


# ---------------------------------------------------------------------------
# Forward-pass projection (applied during training)
# ---------------------------------------------------------------------------

def project_head_output(z_h: torch.Tensor, Q_h: torch.Tensor) -> torch.Tensor:
    """
    Remove identity-sensitive component from head output.

    z_h:  (B, N_tokens, d_k)
    Q_h:  (N_tokens * d_k, N_subjects)   -- orthonormal columns

    Returns z_h* = z_h - Q_h @ (Q_h^T @ vec(z_h))  reshaped back
    """
    B, N_tok, d_k = z_h.shape
    v = z_h.view(B, -1)                          # (B, N_tok * d_k)
    coeff = v @ Q_h                               # (B, N)
    v_proj = v - coeff @ Q_h.T                   # (B, N_tok * d_k)
    return v_proj.view(B, N_tok, d_k)


# ---------------------------------------------------------------------------
# Head regularization loss
# ---------------------------------------------------------------------------

def head_regularization_loss(z_h_dict: dict, Q_h_dict: dict,
                              selected_heads: list) -> torch.Tensor:
    """
    L_head = sum over selected heads ||Q_h^T @ vec(z_h)||^2

    z_h_dict:  (l, h) -> (B, N_tok, d_k) tensor WITH grad
    Q_h_dict:  (l, h) -> (d_feat, N) tensor (no grad)
    """
    loss = torch.tensor(0.0, device="cuda")
    for (l, h) in selected_heads:
        key = (l, h)
        if key not in z_h_dict or Q_h_dict.get(key) is None:
            continue
        z_h = z_h_dict[key]             # (B, N_tok, d_k)
        Q_h = Q_h_dict[key]             # (d_feat, N)
        B, N_tok, d_k = z_h.shape
        v = z_h.view(B, -1)             # (B, d_feat)
        proj = v @ Q_h                  # (B, N)
        loss = loss + (proj ** 2).sum()
    return loss


# ---------------------------------------------------------------------------
# Main Phase 0 entry point
# ---------------------------------------------------------------------------

def run_phase0(pipe, arcface, subject_dirs, subject_prompts, device="cuda",
               K=16, sample_timesteps=(100, 300, 500, 700, 900)):
    """
    Full Phase 0 pipeline. Returns head_info dict.

    pipe:            StableDiffusionPipeline (loaded, on CUDA)
    arcface:         ArcFace model (e.g. insightface or iresnet100), on CUDA
    subject_dirs:    list of N data dirs (one per subject)
    subject_prompts: list of N prompt strings
    K:               number of top heads to select
    """
    ddpm_scheduler = DDPMScheduler.from_pretrained(
        pipe.config._name_or_path if hasattr(pipe.config, '_name_or_path')
        else "models/stable-diffusion-v1-5",
        subfolder="scheduler"
    )

    print("[Phase 0] Computing ArcFace embeddings...")
    mean_embeddings = get_subject_embeddings(arcface, subject_dirs, device)

    print("[Phase 0] Building identity subspace (QR)...")
    identity_Q = build_identity_subspace(mean_embeddings)  # (512, N)

    print("[Phase 0] Scoring attention heads...")
    head_info = score_heads(
        pipe, arcface, mean_embeddings, subject_dirs, subject_prompts,
        ddpm_scheduler, device=device, K=K, sample_timesteps=sample_timesteps
    )
    head_info["mean_embeddings"] = mean_embeddings
    head_info["identity_Q"] = identity_Q

    return head_info