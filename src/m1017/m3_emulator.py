"""m3_emulator.py — CPU (numpy) emulator of the numerics of mamba_ssm's Mamba-3 SISO Triton forward
(`mamba3_siso_combined` -> `angle_dt_fwd` + `mamba3_siso_fwd_kernel`, state-spaces/mamba @ e9594ce).

Purpose (issue state-spaces/mamba#1017): separate *where* the kernel-vs-reference deviation comes from.
Everything is computed in float64 EXCEPT at the exact places where the kernel rounds to bf16 / TF32.
Each rounding point is an independent flag, so a source can be switched on alone or switched off alone.

Rounding points (names used as flags), with the kernel line they model:
  in_qkv  : mamba3_siso_combined casts Q, K, V, Trap, Z to bf16 regardless of the caller's dtype
  in_ang  : mamba3_siso_combined casts Angles to bf16
  ang_store: angle_dt_fwd allocates its output with empty_like(angle) -> the cumulative angle theta_t in [0, 2pi)
            (and the returned final angle state) are stored in bf16 (ulp 2^-5 rad for theta >= 4), then reloaded
  qkpre   : q+bias / k+bias rounded to bf16 (only when the biases are bf16: bf16+bf16 -> bf16)
  qkdot   : diagonal term tl.dot(q*k, ones): bf16 products + bf16 result (bf16 bias) or TF32 products (fp32 bias)
  qkrot   : `.to(q_pre_block.dtype)` after the rotary (only when the biases are bf16)
  qkstore : Q_store / K_store are bf16 buffers (Q.dtype) -> rotated q and scale*rotated k rounded to bf16
  state   : tl.dot(q_block, tl.trans(acc_ssm_states).to(q_block.dtype))  -> fp32 state rounded to bf16
  s       : tl.dot(s_block.to(v_block.dtype), v_block)                 -> intra-chunk scores rounded to bf16
  vdecay  : tl.dot(tl.trans(v_block).to(k_block.dtype), k_block)       -> v*exp(decay) rounded to bf16
  out     : Out = torch.empty(..., dtype=V.dtype) (bf16)               -> output rounded to bf16
Not modelled (measured on the GPU instead): tanh.approx/cos.approx/sin.approx, ex2.approx, fp32 accumulation order.

No torch needed. `reference(...)` = all flags off = exact math of the kernel (float64);
`reference_sequential(...)` is an independent O(T) recurrence used to validate it.
"""
import math
import numpy as np

ALL_FLAGS = ("in_qkv", "in_ang", "ang_store", "qkpre", "qkdot", "qkrot", "qkstore", "state", "s", "vdecay", "out")
GROUPS = {
    "inputs cast to bf16 (combined)": ("in_qkv", "in_ang"),
    "cumulative rotary angle stored in bf16 (angle_dt_fwd output)": ("ang_store",),
    "Q/K preprocessing (bias, qk-dot, rotary, Q/K store)": ("qkpre", "qkdot", "qkrot", "qkstore"),
    "bf16 operands of the chunk matmuls (state, scores, v*decay)": ("state", "s", "vdecay"),
    "bf16 output buffer": ("out",),
}


# ----------------------------------------------------------------------------- rounding helpers
def f32(x):
    return np.asarray(x, dtype=np.float64).astype(np.float32).astype(np.float64)


def bf16(x):
    """round-to-nearest-even float64 -> (float32 ->) bfloat16, returned as float64"""
    a = np.ascontiguousarray(np.asarray(x, dtype=np.float64).astype(np.float32))
    u = a.view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32).astype(np.float64).reshape(a.shape)


def tf32(x):
    """cvt.rna.tf32.f32 (what Triton uses for fp32 tl.dot with input_precision='tf32'): 10 mantissa bits, ties away"""
    a = np.ascontiguousarray(np.asarray(x, dtype=np.float64).astype(np.float32))
    u = a.view(np.uint32).astype(np.uint64)
    u = (u + 0x1000) & 0xFFFFE000
    return u.astype(np.uint32).view(np.float32).astype(np.float64).reshape(a.shape)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)


# ----------------------------------------------------------------------------- inputs (mirror of create_mamba3_siso_inputs)
def make_inputs(seed, batch, seqlen, nheads, nheads_qk, headdim_qk, headdim_v, dtype, has_D, has_Z, has_states):
    """dtype in {'fp32','bf16'}: the dtype the *caller* uses (as in tests/ops/triton/test_mamba3_siso.py).
    Statistically identical to the torch generator (not bit-identical: different RNG)."""
    rng = np.random.default_rng(seed)
    cast = bf16 if dtype == "bf16" else f32
    eps = 2.0 ** -7 if dtype == "bf16" else float(np.finfo(np.float32).eps)

    def rms(x):
        return cast(x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps))

    Q = rms(cast(rng.standard_normal((batch, seqlen, nheads_qk, headdim_qk))))
    K = rms(cast(rng.standard_normal((batch, seqlen, nheads_qk, headdim_qk))))
    V = cast(rng.standard_normal((batch, seqlen, nheads, headdim_v)))
    a = f32(-rng.uniform(1.0, 16.0, (batch, nheads, seqlen)))
    dt = f32(np.exp(rng.uniform(0, 1, (batch, nheads, seqlen)) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)))
    ADT = f32(a * dt)
    Trap = cast(rng.uniform(0.0, 1.0, (batch, nheads, seqlen)))
    Qb = cast(rng.standard_normal((nheads, headdim_qk)))
    Kb = cast(rng.standard_normal((nheads, headdim_qk)))
    Ang = f32(rng.standard_normal((batch, seqlen, nheads, headdim_qk // 4)))
    D = np.ones(nheads) if has_D else None
    Z = cast(rng.standard_normal((batch, seqlen, nheads, headdim_v))) if has_Z else None
    St = None
    if has_states:
        St = (f32(rng.standard_normal((batch, nheads, headdim_qk // 4))),
              f32(rng.standard_normal((batch, nheads, headdim_v, headdim_qk))),
              f32(rng.standard_normal((batch, nheads, headdim_qk))),
              f32(rng.standard_normal((batch, nheads, headdim_v))))
    return dict(Q=Q, K=K, V=V, ADT=ADT, DT=dt, Trap=Trap, Q_bias=Qb, K_bias=Kb, Angles=Ang, D=D, Z=Z,
                Input_States=St, bias_bf16=(dtype == "bf16"))


# ----------------------------------------------------------------------------- the emulator
def _rotate(x, c, s):
    """interleaved pairs (x[2i], x[2i+1]); c, s already padded to d/2 with (1, 0)"""
    y = np.empty_like(x)
    x0, x1 = x[..., 0::2], x[..., 1::2]
    y[..., 0::2] = x0 * c - x1 * s
    y[..., 1::2] = x0 * s + x1 * c
    return y


def forward(inp, flags=ALL_FLAGS, chunk_size=64, angle_fn=None):
    """Emulated mamba3_siso_combined(..., return_final_states=True).
    flags: subset of ALL_FLAGS that are switched ON.  flags=() gives the exact float64 result.
    angle_fn: optional replacement for tanh(x)*pi (to inject a model of tanh.approx).
    Returns out (B,T,H,dv) and final states (angle, ssm, k, v)."""
    flags = set(flags)
    unknown = flags - set(ALL_FLAGS)
    assert not unknown, unknown
    bias_bf16 = inp["bias_bf16"]
    R = lambda name, x: bf16(x) if name in flags else x

    Q, K, V, Z = inp["Q"], inp["K"], inp["V"], inp["Z"]
    ADT, DT, Trap, Ang = inp["ADT"], inp["DT"], inp["Trap"], inp["Angles"]
    Qb, Kb, D, St = inp["Q_bias"], inp["K_bias"], inp["D"], inp["Input_States"]
    B, T, Hqk, dqk = Q.shape
    H, dv = V.shape[2], V.shape[3]
    da = Ang.shape[-1]
    g = H // Hqk

    # --- mamba3_siso_combined: casts
    Q, K, V, Trap = (R("in_qkv", x) for x in (Q, K, V, Trap))
    Z = R("in_qkv", Z) if Z is not None else None
    Ang = R("in_ang", Ang)

    # (B,H,T,*) layout
    Q = np.repeat(Q, g, axis=2).transpose(0, 2, 1, 3)
    K = np.repeat(K, g, axis=2).transpose(0, 2, 1, 3)
    V = V.transpose(0, 2, 1, 3)
    Zt = Z.transpose(0, 2, 1, 3) if Z is not None else None

    # --- angle_dt_fwd: theta_t = (theta_init + cumsum(tanh(angle) * pi * dt)) mod 2pi
    inc = (angle_fn(Ang) if angle_fn is not None else np.tanh(Ang) * math.pi).transpose(0, 2, 1, 3) * DT[..., None]
    th = np.cumsum(inc, axis=2)
    if St is not None:
        th = th + St[0][:, :, None, :]
    th = np.mod(th, 2 * math.pi)
    th = R("ang_store", th)
    c = np.ones(th.shape[:-1] + (dqk // 2,))
    s_ = np.zeros_like(c)
    c[..., :da], s_[..., :da] = np.cos(th), np.sin(th)

    # --- phase 1: trapezoid coefficients, bias, diagonal qk dot, rotary, Q/K store
    trap = sigmoid(Trap)
    gamma = DT * trap
    shifted = np.zeros_like(DT)
    shifted[..., :-1] = DT[..., 1:] * (1.0 - trap[..., 1:])
    scale = gamma + shifted

    q_pre = Q + Qb[None, :, None, :]
    k_pre = K + Kb[None, :, None, :]
    if bias_bf16:
        q_pre, k_pre = R("qkpre", q_pre), R("qkpre", k_pre)
        prod = R("qkdot", q_pre * k_pre)
        qkd = R("qkdot", prod.sum(-1))
    else:
        prod = q_pre * k_pre
        if "qkdot" in flags:
            prod = tf32(prod)
        qkd = prod.sum(-1)
    qkd = qkd * gamma

    k_rot = _rotate(k_pre, c, s_)
    q_rot = _rotate(q_pre, c, s_)
    if bias_bf16:
        k_rot, q_rot = R("qkrot", k_rot), R("qkrot", q_rot)
    k_final = k_rot[:, :, -1, :].copy()
    k_st = R("qkstore", k_rot * scale[..., None])
    q_st = R("qkstore", q_rot)

    # --- phase 2: chunked recurrence
    if St is not None:
        S = St[1] + St[3][..., :, None] * St[2][..., None, :] * (DT[..., 0] * (1.0 - trap[..., 0]))[..., None, None]
    else:
        S = np.zeros((B, H, dv, dqk))
    Dv = D[None, :, None] if D is not None else 0.0
    out = np.empty((B, H, T, dv))
    for c0 in range(0, T, chunk_size):
        sl = slice(c0, min(c0 + chunk_size, T))
        L = sl.stop - sl.start
        dcs = np.cumsum(ADT[..., sl], axis=-1)
        last = dcs[..., -1]
        rev = last[..., None] - dcs
        q, k, v = q_st[:, :, sl], k_st[:, :, sl], V[:, :, sl]
        o = np.einsum("bhtd,bhvd->bhtv", q, R("state", S)) * np.exp(dcs)[..., None]
        sc = np.einsum("bhtd,bhsd->bhts", q, k) * np.exp(np.minimum(dcs[..., :, None] - dcs[..., None, :], 0.0))
        sc = sc * np.tril(np.ones((L, L)), -1)
        o = o + np.einsum("bhts,bhsv->bhtv", R("s", sc), v)
        o = o + (Dv + qkd[:, :, sl])[..., None] * v
        if Zt is not None:
            o = o * silu(Zt[:, :, sl])
        out[:, :, sl] = R("out", o)
        vs = R("vdecay", v * np.exp(rev)[..., None])
        S = S * np.exp(last)[..., None, None] + np.einsum("bhtv,bhtd->bhvd", vs, k)
    return out.transpose(0, 2, 1, 3), (th[:, :, -1, :], S, k_final, V[:, :, -1, :])


def reference(inp, chunk_size=64):
    return forward(inp, flags=(), chunk_size=chunk_size)


def reference_sequential(inp):
    """Independent float64 recurrence (mirror of mamba3_siso_step_ref): S_t = a S_{t-1} + b v_{t-1}k_{t-1}^T + g v_t k_t^T."""
    Q, K, V, Z = inp["Q"], inp["K"], inp["V"], inp["Z"]
    ADT, DT, Trap, Ang = inp["ADT"], inp["DT"], inp["Trap"], inp["Angles"]
    Qb, Kb, D, St = inp["Q_bias"], inp["K_bias"], inp["D"], inp["Input_States"]
    B, T, Hqk, dqk = Q.shape
    H, dv = V.shape[2], V.shape[3]
    da = Ang.shape[-1]
    g = H // Hqk
    Q = np.repeat(Q, g, axis=2)
    K = np.repeat(K, g, axis=2)
    th = St[0].copy() if St is not None else np.zeros((B, H, da))
    S = St[1].copy() if St is not None else np.zeros((B, H, dv, dqk))
    kp = St[2].copy() if St is not None else np.zeros((B, H, dqk))
    vp = St[3].copy() if St is not None else np.zeros((B, H, dv))
    out = np.empty((B, T, H, dv))
    for t in range(T):
        th = np.mod(th + np.tanh(Ang[:, t]) * math.pi * DT[:, :, t][..., None], 2 * math.pi)
        c = np.ones((B, H, dqk // 2)); s_ = np.zeros_like(c)
        c[..., :da], s_[..., :da] = np.cos(th), np.sin(th)
        q = _rotate(Q[:, t] + Qb[None], c, s_)
        k = _rotate(K[:, t] + Kb[None], c, s_)
        v = V[:, t]
        tr = sigmoid(Trap[:, :, t]); al = np.exp(ADT[:, :, t]); dt = DT[:, :, t]
        S = al[..., None, None] * S + ((1 - tr) * dt * al)[..., None, None] * vp[..., :, None] * kp[..., None, :] \
            + (tr * dt)[..., None, None] * v[..., :, None] * k[..., None, :]
        y = np.einsum("bhvd,bhd->bhv", S, q)
        if D is not None:
            y = y + D[None, :, None] * v
        if Z is not None:
            y = y * silu(Z[:, t])
        out[:, t] = y
        kp, vp = k, v
    return out, (th, S, kp, vp)


# ----------------------------------------------------------------------------- metrics
def p95_rel(ker, ref, eps=1e-6, mask=1e-2, p=0.95):
    """exactly tests/ops/triton/test_mamba3_siso.py::relative_error (kthvalue at ceil(p n))"""
    ker = np.asarray(ker, np.float64).astype(np.float32).astype(np.float64).ravel()
    ref = np.asarray(ref, np.float64).astype(np.float32).astype(np.float64).ravel()
    m = np.abs(ref) >= mask
    if not m.any():
        return 0.0
    v = np.sort(np.abs(ker[m] - ref[m]) / (np.abs(ref[m]) + eps))
    k = max(1, min(v.size, int(math.ceil(p * v.size))))
    return float(v[k - 1])


def normwise(ker, ref):
    return float(np.linalg.norm(np.ravel(ker) - np.ravel(ref)) / np.linalg.norm(np.ravel(ref)))


def stats(ker, ref):
    d = np.abs(np.ravel(ker) - np.ravel(ref))
    return dict(p95=p95_rel(ker, ref), norm=normwise(ker, ref), maxabs=float(d.max()), meanabs=float(d.mean()),
                meandiff=float(np.mean(np.ravel(ker) - np.ravel(ref))))
