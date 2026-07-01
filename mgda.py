"""
mgda.py
=======
Multiple Gradient Descent Algorithm (MGDA) via Frank-Wolfe.

Finds the minimum-norm point in the convex hull of N gradient vectors,
which is the Pareto-optimal update direction for N objectives.

Reference: Désidéri (2012), Sener & Koltun (2018 NeurIPS).

Operates on FLATTENED parameter gradient vectors stored on CPU to avoid
OOM (model has ~860M params; 4 copies = 3.4B floats ≈ 13 GB on GPU).
All heavy math happens in float32 on CPU; final combined gradient is
moved to GPU for the optimizer step.
"""

import torch
import numpy as np
from typing import List, Dict


# ---------------------------------------------------------------------------
# Frank-Wolfe QP solver
# ---------------------------------------------------------------------------

def frank_wolfe_qp(grads: List[torch.Tensor],
                   max_iter: int = 20,
                   tol: float = 1e-6) -> torch.Tensor:
    """
    Solve:
        min_{alpha in Delta^N} || sum_i alpha_i * g_i ||^2

    where Delta^N = {alpha : sum=1, alpha_i >= 0}.

    grads: list of N 1-D tensors (same length), on CPU, float32.
    Returns: alpha (N,) tensor of optimal weights.
    """
    N = len(grads)
    if N == 1:
        return torch.ones(1)

    # Gram matrix G[i,j] = <g_i, g_j>  (N x N, cheap for N=4)
    G = torch.zeros(N, N)
    for i in range(N):
        for j in range(i, N):
            val = torch.dot(grads[i], grads[j]).item()
            G[i, j] = val
            G[j, i] = val

    # Initialize uniform
    alpha = torch.full((N,), 1.0 / N)

    for _ in range(max_iter):
        # Current combined gradient (in terms of Gram products)
        # ||g_current||^2 = alpha^T G alpha  (not needed directly)
        # gradient of objective w.r.t. alpha = 2 * G @ alpha
        obj_grad = G @ alpha          # (N,)  = 2*G@alpha / 2

        # Frank-Wolfe: move toward the vertex that minimizes <g_current, g_j>
        j_star = torch.argmin(obj_grad).item()

        # Step size: minimise along the line from alpha to e_{j*}
        # f(gamma) = ||g_current + gamma*(g_{j*} - g_current)||^2
        # = || (1-gamma) g_c + gamma g_{j*} ||^2
        # df/dgamma = 0 => gamma* = (g_c - g_{j*})^T g_c / ||g_c - g_{j*}||^2
        # in Gram form:
        # g_c = sum_i alpha_i g_i  -> ||g_c||^2 = alpha^T G alpha
        # (g_c - g_{j*})^T g_c = alpha^T G alpha - (G alpha)[j*]
        # ||g_c - g_{j*}||^2  = alpha^T G alpha - 2*(G alpha)[j*] + G[j*,j*]

        g_c_sq = (alpha @ G @ alpha).item()
        cross = obj_grad[j_star].item()
        diff_sq = g_c_sq - 2 * cross + G[j_star, j_star].item()

        if diff_sq < 1e-12:
            break

        gamma = (g_c_sq - cross) / diff_sq
        gamma = float(np.clip(gamma, 0.0, 1.0))

        e_j = torch.zeros(N)
        e_j[j_star] = 1.0
        alpha_new = (1 - gamma) * alpha + gamma * e_j

        if (alpha_new - alpha).norm().item() < tol:
            alpha = alpha_new
            break
        alpha = alpha_new

    return alpha   # (N,) sums to 1, all >= 0


# ---------------------------------------------------------------------------
# Gradient extraction / combination utilities
# ---------------------------------------------------------------------------

def extract_flat_grad(model: torch.nn.Module) -> torch.Tensor:
    """
    Flatten all .grad tensors of model into a single CPU float32 vector.
    Parameters with None grad get zeros.
    """
    parts = []
    for p in model.parameters():
        if p.grad is not None:
            parts.append(p.grad.detach().cpu().float().view(-1))
        else:
            parts.append(torch.zeros(p.numel()))
    return torch.cat(parts)


def apply_flat_grad(model: torch.nn.Module, flat_grad: torch.Tensor,
                    device: str = "cuda"):
    """
    Write a flat gradient vector back into model.parameters()[*].grad.
    """
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        chunk = flat_grad[offset: offset + numel].to(device).view(p.shape)
        p.grad = chunk.to(p.dtype)
        offset += numel


def combine_gradients_mgda(flat_grads: List[torch.Tensor],
                            model: torch.nn.Module,
                            device: str = "cuda") -> torch.Tensor:
    """
    Given a list of N flat gradient vectors (CPU), run MGDA to find
    optimal alpha, return the combined flat gradient (CPU).

    Also writes the result into model.grad for immediate optimizer use.
    """
    alpha = frank_wolfe_qp(flat_grads)   # (N,)
    print(f"  [MGDA] alpha = {alpha.numpy().round(3)}")

    combined = torch.zeros_like(flat_grads[0])
    for i, g in enumerate(flat_grads):
        combined += alpha[i].item() * g

    return combined, alpha


# ---------------------------------------------------------------------------
# Per-subject loss computation helper
# ---------------------------------------------------------------------------

def compute_subject_identity_loss(e_gen: torch.Tensor,
                                   mean_emb: torch.Tensor,
                                   tau: float = 0.3) -> torch.Tensor:
    """
    Hinge identity suppression loss for one subject.
    e_gen:    (512,) unit vector of generated image (requires grad)
    mean_emb: (512,) unit vector of subject (no grad)
    tau:      similarity threshold below which loss = 0
    """
    sim = torch.dot(F.normalize(e_gen, dim=0), mean_emb.detach())
    return torch.clamp(sim - tau, min=0.0)


# Make F available (imported in caller but defined here for completeness)
import torch.nn.functional as F