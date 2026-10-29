"""ref_torch64.py — differentiable float64 reference for mamba3_siso_combined (torch, runs on CPU or CUDA).

Chunk-free quadratic ("attention") form of the exponential-trapezoidal recurrence
    S_t = a_t S_{t-1} + (1-l_t) dt_t a_t v_{t-1} k_{t-1}^T + l_t dt_t v_t k_t^T,   y_t = S_t q_t (+ D v_t), out = y * silu(z)
with a_t = exp(ADT_t), l_t = sigmoid(Trap_t), q/k rotated by theta_t = theta_init + cumsum(tanh(Angles) * pi * DT).
Same signature/shapes/returns as mamba3_siso_combined(..., return_final_states=True). Memory O(B H T^2): use T <= ~1k.
"""
import math
import torch
import torch.nn.functional as F


def _rot(x, c, s):
    x0, x1 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x0 * c - x1 * s, x0 * s + x1 * c], dim=-1).flatten(-2)


def mamba3_siso_ref64(Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, D=None, Z=None, Input_States=None, keep=None):
    """keep: optional dict; receives keep['theta'] = cumulative angle (B,H,T,da) with retain_grad(), so that after
    backward keep['theta'].grad = dL/dtheta_t (includes the path through the returned final angle state)."""
    d = lambda x: None if x is None else x.to(torch.float64)
    Q, K, V, ADT, DT, Trap, Qb, Kb, Ang, D, Z = map(d, (Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, D, Z))
    B, T, Hqk, dqk = Q.shape
    H, dv = V.shape[2], V.shape[3]
    da = Ang.shape[-1]
    g = H // Hqk
    Q = Q.repeat_interleave(g, dim=2).transpose(1, 2)
    K = K.repeat_interleave(g, dim=2).transpose(1, 2)
    Vt = V.transpose(1, 2)
    inc = (torch.tanh(Ang) * math.pi).transpose(1, 2) * DT[..., None]
    th = torch.cumsum(inc, dim=2)
    if Input_States is not None:
        st = [d(x) for x in Input_States]
        th = th + st[0][:, :, None, :]
    if keep is not None and th.requires_grad:
        th.retain_grad()
        keep["theta"] = th
    pad = dqk // 2 - da
    c = torch.cat([torch.cos(th), th.new_ones(th.shape[:-1] + (pad,))], -1)
    s = torch.cat([torch.sin(th), th.new_zeros(th.shape[:-1] + (pad,))], -1)
    q = _rot(Q + Qb[None, :, None, :], c, s)
    k = _rot(K + Kb[None, :, None, :], c, s)
    trap = torch.sigmoid(Trap)
    gamma = DT * trap
    shifted = F.pad(DT[..., 1:] * (1 - trap[..., 1:]), (0, 1))
    scale = gamma + shifted
    cs = torch.cumsum(ADT, dim=-1)
    strict = torch.tril(torch.ones(T, T, dtype=torch.bool, device=Q.device), -1)
    diff = (cs[..., :, None] - cs[..., None, :]).masked_fill(~strict, 0.0)
    W = torch.where(strict, torch.exp(diff), torch.zeros_like(diff)) * scale[..., None, :] + torch.diag_embed(gamma)
    y = (W * (q @ k.transpose(-1, -2))) @ Vt
    S0 = None
    if Input_States is not None:
        S0 = st[1] + st[3][..., :, None] * st[2][..., None, :] * (DT[..., 0] * (1 - trap[..., 0]))[..., None, None]
        y = y + torch.exp(cs)[..., None] * (q @ S0.transpose(-1, -2))
    if D is not None:
        y = y + D[None, :, None, None] * Vt
    if Z is not None:
        Zt = Z.transpose(1, 2)
        y = y * Zt * torch.sigmoid(Zt)
    out = y.transpose(1, 2)
    wl = torch.exp(cs[..., -1:] - cs) * scale
    S = torch.einsum("bht,bhtv,bhtd->bhvd", wl, Vt, k)
    if S0 is not None:
        S = S + torch.exp(cs[..., -1])[..., None, None] * S0
    th_final = torch.remainder(th[:, :, -1, :], 2 * math.pi)
    return out, th_final, S, k[:, :, -1, :], Vt[:, :, -1, :]
