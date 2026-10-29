"""CPU emulation of the error mechanism (independent of the authors' GPU code).

Exact FP64 forward/backward of the rotary recurrence (output-only loss, zero initial state).
Local angle terms g_t are then recomputed from BF16-rounded operands (S_t, dL/dS_t, q_t, k_t, v_t,
ybar_t) with FP32 accumulation, mimicking a BF16 kernel. Compare:
  suffix : G_u = sum_{t>=u} g^_t                      (global reverse cumsum, FP32 and FP64 accumulators)
  bound  : G_u = sum_{t=u}^{b-1} g^_t + B^_b          (chunk-local sum + boundary contraction, kappa = 0)
with B^_b computed from BF16-rounded X_b and P_b.  Prediction: suffix error ~ sqrt(T), boundary flat.
"""
import numpy as np, json, sys


def bf16(x):
    x = np.asarray(x, dtype=np.float32)
    b = x.view(np.uint32).astype(np.uint64)
    rounding = ((b >> 16) & 1) + 0x7FFF
    b = ((b + rounding) >> 16) << 16
    return b.astype(np.uint32).view(np.float32)


def rot(x, th):
    y = x.copy(); c, s = np.cos(th), np.sin(th)
    y[..., 0::2] = c * x[..., 0::2] - s * x[..., 1::2]
    y[..., 1::2] = s * x[..., 0::2] + c * x[..., 1::2]
    return y


def Jr(X):  # X J (pairs)
    Y = np.empty_like(X); Y[..., 0::2] = X[..., 1::2]; Y[..., 1::2] = -X[..., 0::2]; return Y


def Jv(k):  # J k
    y = np.empty_like(k); y[..., 0::2] = -k[..., 1::2]; y[..., 1::2] = k[..., 0::2]; return y


def run(T, P=16, N=16, chunk=64, seed=0):
    rng = np.random.default_rng(seed)
    q0 = rng.standard_normal((T, N)); k0 = rng.standard_normal((T, N))
    q0 /= np.linalg.norm(q0, axis=1, keepdims=True); k0 /= np.linalg.norm(k0, axis=1, keepdims=True)
    v = rng.standard_normal((T, P))
    d = np.exp(rng.uniform(np.log(1e-3), np.log(1e-1), T))
    A = -rng.uniform(1, 16, T)
    a = np.exp(A * d); lam = 1 / (1 + np.exp(-rng.standard_normal(T)))
    beta = d * (1 - lam); gamma = d * lam
    rate = rng.standard_normal((T, N // 2))
    theta = np.cumsum(np.pi * d[:, None] * np.tanh(rate), 0)
    q = rot(q0, np.repeat(theta, 1, 1)) if False else rot(q0, theta)
    k = rot(k0, theta)
    ybar = rng.standard_normal((T, P))
    # forward
    S = np.zeros((T, P, N)); X = np.zeros((T, P, N)); Sprev = np.zeros((P, N)); kp = np.zeros(N); vp = np.zeros(P)
    for t in range(T):
        X[t] = Sprev + beta[t] * np.outer(vp, kp)
        S[t] = a[t] * X[t] + gamma[t] * np.outer(v[t], k[t])
        Sprev, kp, vp = S[t], k[t], v[t]
    # backward: dS_t = ybar q^T + dX_{t+1};  dX_t = a_t dS_t
    dS = np.zeros((T, P, N)); dX = np.zeros((T, P, N)); nxt = np.zeros((P, N))
    for t in reversed(range(T)):
        dS[t] = np.outer(ybar[t], q[t]) + nxt
        dX[t] = a[t] * dS[t]
        nxt = dX[t]
    def local_terms(S_, dS_, dX_, q_, k_, v_, ybar_):
        # g_t = ybar^T S J q (query path) + v^T C J k (key path), C = gamma dS_t + beta_{t+1} dX_{t+1}
        dXn = np.concatenate([dX_[1:], np.zeros((1, P, N))], 0)
        bnext = np.concatenate([beta[1:], [0.0]])
        C = gamma[:, None, None] * dS_ + bnext[:, None, None] * dXn
        Sy = np.einsum('tp,tpn->tn', ybar_, S_)            # S^T ybar
        Cv = np.einsum('tp,tpn->tn', v_, C)                # C^T v
        Jq = Jv(q_); Jk = Jv(k_)
        gq = (Sy * Jq).reshape(T, N // 2, 2).sum(-1)
        gk = (Cv * Jk).reshape(T, N // 2, 2).sum(-1)
        return gq + gk
    g = local_terms(S, dS, dX, q, k, v, ybar)
    G = np.flip(np.cumsum(np.flip(g, 0), 0), 0)
    Bx = (dX * Jr(X)).reshape(T, P, N // 2, 2).sum((1, 3))   # B_u = <P_u, X_u J>
    res_identity = np.abs(Bx - G).max() / np.abs(G).max()      # output-only, zero init => kappa = 0
    # "BF16 kernel": round operands, FP32 accumulate
    f32 = np.float32
    gh = local_terms(bf16(S).astype(f32), bf16(dS).astype(f32), bf16(dX).astype(f32), bf16(q).astype(f32),
                     bf16(k).astype(f32), bf16(v).astype(f32), bf16(ybar).astype(f32)).astype(f32)
    Gs32 = np.flip(np.cumsum(np.flip(gh, 0), 0, dtype=f32), 0)
    Gs64 = np.flip(np.cumsum(np.flip(gh.astype(np.float64), 0), 0), 0)   # exact accumulator, same terms
    Bh = (bf16(dX).astype(f32) * Jr(bf16(X).astype(f32))).reshape(T, P, N // 2, 2).sum((1, 3), dtype=f32)
    Gb = np.empty_like(Gs32)
    for u in range(T):
        b = (u // chunk + 1) * chunk
        Gb[u] = gh[u:min(b, T)].sum(0, dtype=f32) + (Bh[b] if b < T else 0)
    rel = lambda E: float(np.linalg.norm(E - G) / np.linalg.norm(G))
    return dict(T=T, identity=float(res_identity), suffix_fp32=rel(Gs32), suffix_exactsum=rel(Gs64), boundary=rel(Gb),
                local_term_rel=float(np.linalg.norm(gh - g) / np.linalg.norm(g)),
                cancellation=float(np.abs(g).sum(0).mean() / (np.abs(G).mean() + 1e-300)))


if __name__ == '__main__':
    out = []
    for T in (256, 512, 1024, 2048, 4096, 8192):
        for seed in (0, 1, 2):
            r = run(T, seed=seed); out.append(r); print(r, flush=True)
    Ts = np.array([r['T'] for r in out], float)
    for key in ('suffix_fp32', 'suffix_exactsum', 'boundary'):
        e = np.array([r[key] for r in out]); slope = np.polyfit(np.log(Ts), np.log(e), 1)[0]
        print(key, 'log-log slope vs T = %.3f' % slope)
    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else 'bf16_emulation.json', 'w'), indent=1)
