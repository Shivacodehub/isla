"""
diagnose_identity.py
====================
Measures WHERE identity information lives in a UNet checkpoint.

For each outer iteration checkpoint, computes:
  1. ||Q_h^T z_h|| for protected heads  -> identity inside projected heads
  2. ||Q_h^T z_h|| for unprotected heads -> identity migration to other heads
  3. ArcFace similarity of generated image -> overall identity leakage

This tells us:
  - If protected heads show low values but DINO is high
    -> identity migrated to unprotected heads or FFN/residual
  - If protected heads show rising values
    -> projection is failing (Q_h stale or hooks broken)
  - If all heads show low values but DINO is high
    -> identity is stored in FFN/cross-attn, not self-attn at all

Usage:
  python diagnose_identity.py \
    --pretrained_model_name_or_path models/stable-diffusion-v1-5 \
    --checkpoint_dir experiments/isla_realface \
    --subject_data_dirs data/person1/set_B data/person2/set_B \
                        data/person3/set_B data/person4/set_B \
    --subject_prompts "a photo of sks1 person" ... \
    --head_info_pkl experiments/isla_realface/head_info_004.pkl
"""

import os
import pickle
import torch
import torch.nn.functional as F
import numpy as np
from argparse import ArgumentParser
from diffusers import StableDiffusionPipeline, DDPMScheduler
from PIL import Image
from torchvision import transforms
from collections import defaultdict

from arcface_torch import load_arcface_torch
from identity_heads import ISLAHookManager


# ===========================================================================
# Utilities
# ===========================================================================

def load_images(data_dir, size=112):
    tf = transforms.Compose([
        transforms.Resize(size), transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)])
    imgs = []
    for f in os.listdir(data_dir):
        if f.lower().endswith(('.jpg','.jpeg','.png')):
            imgs.append(tf(Image.open(os.path.join(data_dir,f)).convert("RGB")))
    return torch.stack(imgs)  # (N, 3, 112, 112)


def encode_prompt(tokenizer, text_encoder, prompt):
    inputs = tokenizer(prompt, padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt").to("cuda")
    return text_encoder(inputs.input_ids)[0]


# ===========================================================================
# Measurement 1: ||Q_h^T z_h|| for all heads
# ===========================================================================

class AllHeadMonitor:
    """Captures z_h for ALL attention heads (both protected and unprotected)."""
    def __init__(self, unet):
        self.activations = {}  # (layer, head) -> tensor
        self.hooks = []
        self._register(unet)

    def _make_hook(self, layer_idx, module):
        def hook(mod, input, output):
            hidden = input[0]
            B, N, d = hidden.shape
            H = mod.heads
            d_k = d // H
            v = mod.to_v(hidden)
            q = mod.to_q(hidden)
            k = mod.to_k(hidden)
            def split(x):
                return x.view(B, N, H, d_k).permute(0,2,1,3).reshape(B*H, N, d_k)
            q_h = split(q); k_h = split(k); v_h = split(v)
            attn = torch.softmax(
                torch.bmm(q_h, k_h.transpose(-1,-2)) / (d_k**0.5), dim=-1)
            z = torch.bmm(attn, v_h).view(B, H, N, d_k)
            for h in range(H):
                self.activations[(layer_idx, h)] = z[:,h,:,:].detach()
        return hook

    def _register(self, unet):
        idx = 0
        for name, mod in unet.named_modules():
            if "attn1" in name and hasattr(mod, "to_q") and hasattr(mod, "heads"):
                self.hooks.append(mod.register_forward_hook(self._make_hook(idx, mod)))
                idx += 1

    def clear(self): self.activations.clear()

    def remove(self):
        for h in self.hooks: h.remove()
        self.hooks.clear()


def measure_identity_in_heads(unet, Q_h_dict, selected_heads, all_head_keys,
                               z_t, t, prompt_emb):
    """
    Returns:
      protected_scores:   dict (l,h) -> ||Q_h^T z_h|| (protected heads)
      unprotected_scores: dict (l,h) -> ||Q_h^T z_h|| (unprotected, using same Q_h)
      all_activation_norms: dict (l,h) -> ||z_h|| (all heads)
    """
    monitor = AllHeadMonitor(unet)
    with torch.no_grad():
        unet(z_t, t, encoder_hidden_states=prompt_emb, return_dict=False)

    protected_scores = {}
    unprotected_scores = {}
    all_norms = {}

    selected_set = set(map(tuple, selected_heads))

    for key, z_h in monitor.activations.items():
        # z_h: (B, N_tok, d_k)
        B, N_tok, d_k = z_h.shape
        v = z_h.view(B, -1)  # (B, N_tok*d_k)
        all_norms[key] = v.norm().item()

        # Measure identity component using Q_h from protected heads
        # For unprotected heads, use the nearest protected head's Q_h
        if key in Q_h_dict and Q_h_dict[key] is not None:
            Q_h = Q_h_dict[key]  # (d_feat, N)
            # Project v onto Q_h subspace
            if Q_h.shape[0] == v.shape[1]:
                proj = (v @ Q_h)  # (B, N_subj)
                score = proj.norm().item()
            else:
                # Dimension mismatch — use truncated projection
                min_d = min(Q_h.shape[0], v.shape[1])
                proj = (v[:, :min_d] @ Q_h[:min_d])
                score = proj.norm().item()

            if key in selected_set:
                protected_scores[key] = score
            else:
                unprotected_scores[key] = score

    monitor.remove()
    return protected_scores, unprotected_scores, all_norms


# ===========================================================================
# Measurement 2: ArcFace similarity of generated image
# ===========================================================================

@torch.no_grad()
def measure_generation_identity(pipe, arcface, subject_embs,
                                 subject_prompts, tokenizer, text_encoder,
                                 ddpm_scheduler, num_steps=20):
    """Generate one image per subject and measure ArcFace similarity."""
    sims = []
    num_train_timesteps = ddpm_scheduler.config.num_train_timesteps
    for i, (prompt, e_ref) in enumerate(zip(subject_prompts, subject_embs)):
        # Simple DDIM-style generation (fast, 20 steps)
        z = torch.randn(1, 4, 64, 64, device="cuda")
        pemb = encode_prompt(tokenizer, text_encoder, prompt)
        scheduler = DDPMScheduler.from_config(ddpm_scheduler.config)

        for step_idx in range(num_steps):
            # Clamp to [0, num_train_timesteps - 1]; without this, step_idx=0
            # evaluates to exactly num_train_timesteps (e.g. 1000), which is
            # out of bounds for alphas_cumprod (valid indices 0..999).
            t_val = min(num_train_timesteps - 1,
                        int(num_train_timesteps * (1 - step_idx/num_steps)))
            t = torch.tensor([t_val], device="cuda").long()
            eps_pred = pipe.unet(z, t,
                encoder_hidden_states=pemb, return_dict=False)[0]
            alpha_bar = ddpm_scheduler.alphas_cumprod[t_val].to("cuda")
            z = (z - (1-alpha_bar).sqrt()*eps_pred) / alpha_bar.sqrt().clamp(1e-8)
            if step_idx < num_steps - 1:
                t_next = min(num_train_timesteps - 1,
                             int(num_train_timesteps * (1 - (step_idx+1)/num_steps)))
                alpha_bar_next = ddpm_scheduler.alphas_cumprod[t_next].to("cuda")
                z = alpha_bar_next.sqrt()*z + (1-alpha_bar_next).sqrt()*torch.randn_like(z)

        img = pipe.vae.decode(z/pipe.vae.config.scaling_factor).sample
        img_face = F.interpolate((img*0.5+0.5).clamp(0,1), size=112,
                                  mode="bilinear", align_corners=False)
        e_gen = arcface((img_face-0.5)/0.5)
        e_gen = F.normalize(e_gen, dim=1).squeeze(0)
        sim = torch.dot(e_gen, e_ref.squeeze(0)).item()
        sims.append(sim)
        print(f"  Subject {i+1}: ArcFace sim = {sim:.4f}")
    return sims


# ===========================================================================
# Main diagnosis
# ===========================================================================

def parse_args():
    p = ArgumentParser()
    p.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    p.add_argument("--checkpoint_dir", type=str, required=True,
                   help="Folder containing unet_XXX.pt checkpoints")
    p.add_argument("--head_info_pkl", type=str, required=True,
                   help="Path to head_info_XXX.pkl from training")
    p.add_argument("--subject_data_dirs", nargs="+", required=True)
    p.add_argument("--subject_prompts", nargs="+", required=True)
    p.add_argument("--phase0_timesteps", nargs="+", type=int,
                   default=[100, 300, 500])
    return p.parse_args()


def main(args):
    print("[INFO] Loading pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        safety_checker=None, torch_dtype=torch.float32).to("cuda")
    pipe.safety_checker = None

    ddpm_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")

    print("[INFO] Loading ArcFace...")
    arcface = load_arcface_torch(device="cuda")

    print("[INFO] Loading head info...")
    with open(args.head_info_pkl, "rb") as f:
        head_info = pickle.load(f)
    selected_heads = head_info["selected_heads"]
    Q_h_dict = head_info.get("Q_h_dict", {})

    print(f"[INFO] Selected heads: {selected_heads[:5]} ...")
    print(f"[DEBUG] Q_h_dict has {len(Q_h_dict)} keys, sample: "
          f"{list(Q_h_dict.keys())[:5]}")

    # Compute reference ArcFace embeddings
    print("[INFO] Computing reference embeddings...")
    subject_embs = []
    tf_face = transforms.Compose([
        transforms.Resize(112), transforms.CenterCrop(112),
        transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
    for data_dir in args.subject_data_dirs:
        imgs = []
        for f in os.listdir(data_dir):
            if f.lower().endswith(('.jpg','.jpeg','.png')):
                imgs.append(tf_face(Image.open(
                    os.path.join(data_dir,f)).convert("RGB")))
        imgs_t = torch.stack(imgs)
        with torch.no_grad():
            embs = arcface(imgs_t)
        mean_emb = F.normalize(embs.mean(0, keepdim=True), dim=1)
        subject_embs.append(mean_emb)
        print(f"  {data_dir}: {len(imgs)} images, emb norm={mean_emb.norm():.3f}")

    # Find all checkpoints
    ckpts = sorted([f for f in os.listdir(args.checkpoint_dir)
                    if f.startswith("unet_") and f.endswith(".pt")])
    print(f"\n[INFO] Found {len(ckpts)} checkpoints: {ckpts}")

    print("\n" + "="*70)
    print("DIAGNOSIS RESULTS")
    print("="*70)
    print(f"{'Checkpoint':<20} {'Prot.Heads||Q^Tz||':<22} "
          f"{'UnProt.Heads||Q^Tz||':<22} {'ArcFace sim'}")
    print("-"*70)

    for ckpt_name in ckpts:
        ckpt_path = os.path.join(args.checkpoint_dir, ckpt_name)
        print(f"\nLoading {ckpt_name}...")

        # Load checkpoint into pipe.unet
        state = torch.load(ckpt_path, map_location="cuda")
        pipe.unet.load_state_dict(state)
        pipe.unet.eval()

        # Measure identity in heads across all subjects and timesteps
        prot_scores_all = defaultdict(list)
        unprot_scores_all = defaultdict(list)
        all_head_keys = []

        for si, (data_dir, prompt) in enumerate(
                zip(args.subject_data_dirs, args.subject_prompts)):
            pemb = encode_prompt(pipe.tokenizer, pipe.text_encoder, prompt)

            for t_val in args.phase0_timesteps:
                # Get a real image latent
                imgs = load_images(data_dir, size=512)
                img = imgs[0:1].to("cuda")
                with torch.no_grad():
                    z0 = pipe.vae.encode(img).latent_dist.sample() \
                         * pipe.vae.config.scaling_factor
                alpha_bar = ddpm_scheduler.alphas_cumprod[t_val].to("cuda")
                eps = torch.randn_like(z0)
                z_t = alpha_bar.sqrt()*z0 + (1-alpha_bar).sqrt()*eps
                t = torch.tensor([t_val], device="cuda").long()

                prot, unprot, norms = measure_identity_in_heads(
                    pipe.unet, Q_h_dict, selected_heads,
                    all_head_keys,
                    z_t, t, pemb)
                if not all_head_keys:
                    print(f"[DEBUG] Monitor captured {len(norms)} head keys, "
                          f"sample: {list(norms.keys())[:5]}")
                    overlap = set(norms.keys()) & set(Q_h_dict.keys())
                    print(f"[DEBUG] Overlap between monitor keys and "
                          f"Q_h_dict keys: {len(overlap)}")
                all_head_keys = list(norms.keys())

                for k, v in prot.items():
                    prot_scores_all[k].append(v)
                for k, v in unprot.items():
                    unprot_scores_all[k].append(v)

        # Average scores
        avg_prot = np.mean([np.mean(v) for v in prot_scores_all.values()]) \
                   if prot_scores_all else 0.0
        avg_unprot = np.mean([np.mean(v) for v in unprot_scores_all.values()]) \
                     if unprot_scores_all else 0.0

        # Top-5 worst unprotected heads
        top_unprot = sorted(unprot_scores_all.items(),
                            key=lambda x: np.mean(x[1]), reverse=True)[:5]

        # ArcFace similarity of generated images
        print(f"  Measuring generation identity similarity...")
        sims = measure_generation_identity(
            pipe, arcface, subject_embs, args.subject_prompts,
            pipe.tokenizer, pipe.text_encoder, ddpm_scheduler)
        avg_sim = np.mean(sims)

        print(f"\n  {ckpt_name:<20} "
              f"Prot={avg_prot:.4f}  "
              f"UnProt={avg_unprot:.4f}  "
              f"ArcFace_sim={avg_sim:.4f}")
        print(f"  Top-5 unprotected heads by identity score:")
        for (l,h), scores in top_unprot:
            print(f"    Layer {l:2d} Head {h}: {np.mean(scores):.4f}")

        print(f"\n  DIAGNOSIS:")
        if avg_prot < 0.01 and avg_sim > 0.3:
            print("  -> Identity NOT in protected heads but still leaking")
            print("  -> MIGRATION: identity moved to unprotected heads/FFN")
            print("  -> FIX: increase top_k_heads or add FFN projection")
        elif avg_prot > 0.05 and avg_sim > 0.3:
            print("  -> Identity IS in protected heads and leaking")
            print("  -> PROJECTION FAILING: Q_h stale or hooks not working")
            print("  -> FIX: recompute Q_h more frequently")
        elif avg_sim < 0.3:
            print("  -> GOOD: Identity suppressed (ArcFace sim < 0.3)")
        print("-"*70)


if __name__ == "__main__":
    args = parse_args()
    main(args)