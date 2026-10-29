"""Independent FP64 check of the boundary identity, terminal correction, chunk formula,
chain rule and the supplement's local-coordinate adjoint.

Written from the manuscript equations only (does not import or read the authors' code).
Conventions (manuscript Sec. 2):
  q_t = R(theta_t) q0_t, k_t = R(theta_t) k0_t   (column vectors, pairs (2i,2i+1))
  X_t = S_{t-1} + beta_t v_{t-1} k_{t-1}^T
  S_t = a_t X_t + gamma_t v_t k_t^T,  y_t = S_t q_t
  a_t = exp(ADT_t), lam_t = sigmoid(Trap_t), beta_t = d_t (1-lam_t), gamma_t = d_t lam_t
  delta_{t,i} = pi d_t tanh(A_{t,i}),  theta_t = theta_{-1} + sum_{j<=t} delta_j
"""
import math, json, sys, itertools
import torch

torch.set_default_dtype(torch.float64)
PI = math.pi


def rot_vec(x, th, Na):
    """Apply block rotation R(th) to vector(s) x[..., N] on the first Na pairs."""
    x = x.clone()
    c, s = torch.cos(th), torch.sin(th)
    x0 = x[..., 0:2 * Na:2].clone(); x1 = x[..., 1:2 * Na:2].clone()
    x[..., 0:2 * Na:2] = c * x0 - s * x1
    x[..., 1:2 * Na:2] = s * x0 + c * x1
    return x


def Jright(X, i):
    """X J_i for row-state X (P x N): (XJ)[:,2i] = X[:,2i+1], (XJ)[:,2i+1] = -X[:,2i]."""
    Y = torch.zeros_like(X)
    Y[:, 2 * i] = X[:, 2 * i + 1]
    Y[:, 2 * i + 1] = -X[:, 2 * i]
    return Y


def Jvec(k, i):
    y = torch.zeros_like(k)
    y[2 * i] = -k[2 * i + 1]
    y[2 * i + 1] = k[2 * i]
    return y


def make_inputs(T, P, N, Na, seed, initial):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    inp = dict(
        q0=r(T, N), k0=r(T, N), v=r(T, P), z=r(T, P),
        ADT=-torch.rand(T, generator=g) * 2 - 0.05,
        d=torch.rand(T, generator=g) * 0.5 + 0.05,
        Trap=r(T), A=r(T, Na), D=r(P), theta_init=r(Na),
    )
    if initial:
        inp.update(S_init=r(P, N), k_init=r(N), v_init=r(P))
    else:
        inp.update(S_init=torch.zeros(P, N), k_init=torch.zeros(N), v_init=torch.zeros(P))
    return inp


def forward(inp, Na, theta_override=None, keep=None):
    """Sequential recurrence. theta_override lets cumulative angles be independent leaves."""
    T, N = inp['q0'].shape
    a = torch.exp(inp['ADT']); lam = torch.sigmoid(inp['Trap'])
    beta = inp['d'] * (1 - lam); gamma = inp['d'] * lam
    delta = PI * inp['d'][:, None] * torch.tanh(inp['A'])
    theta = inp['theta_init'][None, :] + torch.cumsum(delta, 0) if theta_override is None else theta_override
    S = inp['S_init']; kprev = inp['k_init']; vprev = inp['v_init']
    outs = []
    Xs = []
    for t in range(T):
        q = rot_vec(inp['q0'][t], theta[t], Na); k = rot_vec(inp['k0'][t], theta[t], Na)
        X = S + beta[t] * torch.outer(vprev, kprev)
        if keep is not None:
            X.retain_grad(); Xs.append(X)
        S = a[t] * X + gamma[t] * torch.outer(inp['v'][t], k)
        y = S @ q
        out = (y + inp['D'] * inp['v'][t]) * torch.nn.functional.silu(inp['z'][t])  # skip + gate
        outs.append(out)
        kprev, vprev = k, inp['v'][t]
    if keep is not None:
        keep['X'] = Xs
    return torch.stack(outs), (S, kprev, vprev, theta[-1]), theta, delta


def loss_fn(out, final, cot, terminal_mask):
    """Nonlinear output loss + optional terminal cotangent terms (bit mask over S,k,v,theta)."""
    L = (cot['W'] * out).sum() + 0.1 * (out ** 2).sum()
    S, k, v, th = final
    if terminal_mask & 1: L = L + (cot['S'] * S).sum() + 0.05 * (S ** 2).sum()
    if terminal_mask & 2: L = L + (cot['k'] * k).sum()
    if terminal_mask & 4: L = L + (cot['v'] * v).sum()
    if terminal_mask & 8: L = L + (cot['th'] * th).sum() + 0.3 * torch.sin(th).sum()
    return L


def make_cot(T, P, N, Na, seed):
    g = torch.Generator().manual_seed(seed + 7777)
    r = lambda *s: torch.randn(*s, generator=g)
    return dict(W=r(T, P), S=r(P, N), k=r(N), v=r(P), th=r(Na))


def check_case(T, P, N, Na, seed, initial, mask, chunk=4):
    inp = make_inputs(T, P, N, Na, seed, initial)
    cot = make_cot(T, P, N, Na, seed)
    # (1) g_t: gradient wrt cumulative angles treated as independent leaves
    with torch.no_grad():
        _, _, theta0, delta0 = forward(inp, Na)
    th_leaf = theta0.clone().requires_grad_(True)
    keep = {}
    inp_g = dict(inp); inp_g['S_init'] = inp['S_init'].clone().requires_grad_(True)  # so X_0 is tracked
    out, final, _, _ = forward(inp_g, Na, theta_override=th_leaf, keep=keep)
    # terminal direct cotangents: make returned finals separate leaves for "direct" partials
    S_f = final[0].detach().clone().requires_grad_(True)
    k_f = final[1].detach().clone().requires_grad_(True)
    v_f = final[2].detach().clone().requires_grad_(True)
    t_f = final[3].detach().clone().requires_grad_(True)
    Lterm = loss_fn(torch.zeros_like(out), (S_f, k_f, v_f, t_f), cot, mask) - loss_fn(torch.zeros_like(out), (S_f.detach()*0, k_f.detach()*0, v_f.detach()*0, t_f.detach()*0), cot, 0)
    L = loss_fn(out, final, cot, mask)
    L.backward()
    g = th_leaf.grad.clone()                     # (T, Na)
    P_adj = [X.grad.clone() for X in keep['X']]   # dL/dX_u (total adjoint of the folded state)
    X_val = [X.detach() for X in keep['X']]
    # direct terminal cotangents
    if Lterm.requires_grad:
        Lterm.backward()
    Sbar = S_f.grad if mask & 1 else torch.zeros(P, N)
    kbar = k_f.grad if mask & 2 else torch.zeros(N)
    thbar = t_f.grad if mask & 8 else torch.zeros(Na)
    S_last, k_last = final[0].detach(), final[1].detach()
    G = torch.flip(torch.cumsum(torch.flip(g, [0]), 0), [0])   # suffix sums
    kappa = torch.stack([-(Sbar * Jright(S_last, i)).sum() + kbar @ Jvec(k_last, i) + thbar[i] for i in range(Na)])
    B = torch.stack([torch.stack([(P_adj[u] * Jright(X_val[u], i)).sum() for i in range(Na)]) for u in range(T)])
    scale = max(1.0, G.abs().max().item())
    res_identity = ((B + kappa[None, :]) - G).abs().max().item() / scale
    res_no_kappa = (B - G).abs().max().item() / scale
    # wrong variant: unfolded state S_{u-1} with its total adjoint dL/dS_{u-1} (includes prefix output)
    # emulate by recomputing: use X_u minus fold term and P_u; and P including readout of u-1
    # chunk formula
    res_chunk = 0.0
    for u in range(T):
        b = (u // chunk + 1) * chunk
        if b >= T:
            continue
        Gc = g[u:b].sum(0) + B[b] + kappa
        res_chunk = max(res_chunk, (Gc - G[u]).abs().max().item() / scale)
    # (2) chain rule: G == dL/d delta (autograd through the real parameterization)
    inp2 = {k_: v_.clone().requires_grad_(k_ in ('A', 'd', 'ADT', 'Trap', 'q0', 'k0', 'v', 'z', 'D', 'S_init', 'k_init', 'v_init', 'theta_init')) for k_, v_ in inp.items()}
    out2, final2, theta2, delta2 = forward(inp2, Na)
    delta2.retain_grad()
    L2 = loss_fn(out2, final2, cot, mask)
    L2.backward()
    res_delta = (delta2.grad - G).abs().max().item() / scale
    dA_formula = PI * inp['d'][:, None] / torch.cosh(inp['A']) ** 2 * G
    res_dA = (inp2['A'].grad - dA_formula).abs().max().item() / max(1.0, inp2['A'].grad.abs().max().item())
    # timestep: total dL/dd minus the rotational part must equal dL/dd with delta frozen
    rot_part = PI * (torch.tanh(inp['A']) * G).sum(1)
    inp3 = {k_: v_.clone() for k_, v_ in inp.items()}
    d_leaf = inp3['d'].clone().requires_grad_(True)
    inp3['d'] = d_leaf
    # freeze delta: compute theta from detached d
    delta_frozen = PI * inp['d'][:, None] * torch.tanh(inp['A'])
    theta_frozen = inp['theta_init'][None, :] + torch.cumsum(delta_frozen, 0)
    out3, final3, _, _ = forward(inp3, Na, theta_override=theta_frozen)
    loss_fn(out3, final3, cot, mask).backward()
    res_dt = (inp2['d'].grad - (d_leaf.grad + rot_part)).abs().max().item() / max(1.0, inp2['d'].grad.abs().max().item())
    return dict(T=T, Na=Na, initial=initial, mask=mask, identity=res_identity, no_kappa=res_no_kappa,
                chunk=res_chunk, delta_chain=res_delta, dA_chain=res_dA, dt_chain=res_dt,
                G0=G[0].abs().max().item(), kappa_norm=kappa.abs().max().item())


def unfolded_counterexample(T=9, P=3, N=6, Na=3, seed=5):
    """Show that using the pre-fold state S_{u-1} and dL/dS_{u-1} breaks the identity."""
    inp = make_inputs(T, P, N, Na, seed, True); cot = make_cot(T, P, N, Na, seed)
    with torch.no_grad():
        _, _, theta0, _ = forward(inp, Na)
    th_leaf = theta0.clone().requires_grad_(True)
    # rebuild forward keeping S_{u-1}
    a = torch.exp(inp['ADT']); lam = torch.sigmoid(inp['Trap']); beta = inp['d'] * (1 - lam); gamma = inp['d'] * lam
    S = inp['S_init'].clone().requires_grad_(True); Ss = [S]; kprev = inp['k_init']; vprev = inp['v_init']; outs = []
    for t in range(T):
        q = rot_vec(inp['q0'][t], th_leaf[t], Na); k = rot_vec(inp['k0'][t], th_leaf[t], Na)
        X = S + beta[t] * torch.outer(vprev, kprev)
        S = a[t] * X + gamma[t] * torch.outer(inp['v'][t], k); S.retain_grad(); Ss.append(S)
        outs.append((S @ q + inp['D'] * inp['v'][t]) * torch.nn.functional.silu(inp['z'][t]))
        kprev, vprev = k, inp['v'][t]
    L = loss_fn(torch.stack(outs), (S, kprev, vprev, th_leaf[-1]), cot, 0); L.backward()
    g = th_leaf.grad; G = torch.flip(torch.cumsum(torch.flip(g, [0]), 0), [0])
    worst = 0.0
    for u in range(1, T):
        Sprev = Ss[u].detach(); Pprev = Ss[u].grad   # S_{u-1} and dL/dS_{u-1} (includes y_{u-1})
        Bw = torch.stack([(Pprev * Jright(Sprev, i)).sum() for i in range(Na)])
        worst = max(worst, (Bw - G[u]).abs().max().item() / max(1.0, G.abs().max().item()))
    return worst


def local_adjoint_check(T=11, P=3, N=6, Na=2, seed=3):
    """Implement the supplement's local-coordinate reverse pass by hand (zero initial state,
    output-only loss, no gate/skip so ybar = dL/dy) and compare all gradients with autograd."""
    inp = make_inputs(T, P, N, Na, seed, False); cot = make_cot(T, P, N, Na, seed)
    # autograd reference on the global-angle formulation, linear output loss W.y (no skip/gate)
    leaves = {k_: inp[k_].clone().requires_grad_(True) for k_ in ('q0', 'k0', 'v', 'ADT', 'd', 'Trap', 'A')}
    a = torch.exp(leaves['ADT']); lam = torch.sigmoid(leaves['Trap']); beta = leaves['d'] * (1 - lam); gamma = leaves['d'] * lam
    delta = PI * leaves['d'][:, None] * torch.tanh(leaves['A']); delta.retain_grad()
    theta = inp['theta_init'][None, :] + torch.cumsum(delta, 0)
    S = torch.zeros(P, N); kprev = torch.zeros(N); vprev = torch.zeros(P); L = 0
    for t in range(T):
        q = rot_vec(leaves['q0'][t], theta[t], Na); k = rot_vec(leaves['k0'][t], theta[t], Na)
        X = S + beta[t] * torch.outer(vprev, kprev)
        S = a[t] * X + gamma[t] * torch.outer(leaves['v'][t], k)
        L = L + cot['W'][t] @ (S @ q)
        kprev, vprev = k, leaves['v'][t]
    L.backward()
    ref = {k_: v_.grad for k_, v_ in leaves.items()}; ref['delta'] = delta.grad
    # ---- hand-written local-coordinate adjoint (supplement eqs) ----
    with torch.no_grad():
        a = torch.exp(inp['ADT']); lam = torch.sigmoid(inp['Trap']); d = inp['d']
        beta = d * (1 - lam); gamma = d * lam
        dl = PI * d[:, None] * torch.tanh(inp['A'])
        q0, k0, v = inp['q0'], inp['k0'], inp['v']
        def rowrot(Xm, dlt):  # X R(delta): [x0 c + x1 s, -x0 s + x1 c]
            Y = Xm.clone(); c, s = torch.cos(dlt), torch.sin(dlt)
            x0 = Xm[:, 0:2 * Na:2]; x1 = Xm[:, 1:2 * Na:2]
            Y[:, 0:2 * Na:2] = x0 * c + x1 * s; Y[:, 1:2 * Na:2] = -x0 * s + x1 * c
            return Y
        def rowrotT(Xm, dlt):
            return rowrot(Xm, -dlt)
        H = [None] * T; F = [None] * T; M = [None] * T
        Hprev = torch.zeros(P, N); kp = torch.zeros(N); vp = torch.zeros(P)
        for t in range(T):
            F[t] = Hprev + beta[t] * torch.outer(vp, kp)
            M[t] = a[t] * rowrot(F[t], dl[t])
            H[t] = M[t] + gamma[t] * torch.outer(v[t], k0[t])
            Hprev, kp, vp = H[t], k0[t], v[t]
        g = {k_: torch.zeros_like(inp[k_]) for k_ in ('q0', 'k0', 'v', 'ADT', 'd', 'Trap')}
        gdelta = torch.zeros(T, Na)
        U = torch.zeros(P, N)  # U_{T-1} = 0
        for t in reversed(range(T)):
            ybar = cot['W'][t]
            Pt = U + torch.outer(ybar, q0[t])
            g['q0'][t] = H[t].T @ ybar
            beta_next = beta[t + 1] if t + 1 < T else 0.0
            C = gamma[t] * Pt + beta_next * U
            g['k0'][t] = C.T @ v[t]
            g['v'][t] = C @ k0[t]
            for i in range(Na):
                gdelta[t, i] = (Pt[:, 2 * i] * M[t][:, 2 * i + 1] - Pt[:, 2 * i + 1] * M[t][:, 2 * i]).sum()
            V = torch.outer(v[t], k0[t])
            W = a[t] * rowrot(torch.outer(v[t - 1], k0[t - 1]) if t > 0 else torch.zeros(P, N), dl[t])
            g['ADT'][t] = (Pt * M[t]).sum()
            d_direct = (Pt * ((1 - lam[t]) * W + lam[t] * V)).sum()
            g['Trap'][t] = d[t] * lam[t] * (1 - lam[t]) * (Pt * (V - W)).sum()
            g['d'][t] = d_direct + PI * (torch.tanh(inp['A'][t]) * gdelta[t]).sum()
            U = a[t] * rowrotT(Pt, dl[t])   # U_{t-1} = a_t P_t R(delta_t)^T
        gA = PI * d[:, None] / torch.cosh(inp['A']) ** 2 * gdelta
    errs = {}
    for k_ in ('q0', 'k0', 'v', 'ADT', 'Trap', 'd'):
        errs[k_] = ((g[k_] - ref[k_]).abs().max() / max(1.0, ref[k_].abs().max())).item()
    errs['delta'] = ((gdelta - ref['delta']).abs().max() / max(1.0, ref['delta'].abs().max())).item()
    errs['A'] = ((gA - ref['A']).abs().max() / max(1.0, ref['A'].abs().max())).item()
    return errs


def fd_check(seed=11):
    """Central finite differences of L w.r.t. delta_u vs B_u + kappa (all terminal terms on)."""
    T, P, N, Na = 7, 3, 6, 2
    inp = make_inputs(T, P, N, Na, seed, True); cot = make_cot(T, P, N, Na, seed)
    def L_of_theta(th):
        out, fin, _, _ = forward(inp, Na, theta_override=th)
        return loss_fn(out, fin, cot, 15)
    with torch.no_grad():
        _, _, theta0, _ = forward(inp, Na)
    worst = 0.0
    th_leaf = theta0.clone().requires_grad_(True)
    out, fin, _, _ = forward(inp, Na, theta_override=th_leaf)
    loss_fn(out, fin, cot, 15).backward()
    G = torch.flip(torch.cumsum(torch.flip(th_leaf.grad, [0]), 0), [0])
    for u in range(T):
        for i in range(Na):
            h = 1e-5
            e = torch.zeros_like(theta0); e[u:, i] = 1.0
            with torch.no_grad():
                fd = (L_of_theta(theta0 + h * e) - L_of_theta(theta0 - h * e)) / (2 * h)
            worst = max(worst, abs(fd.item() - G[u, i].item()) / max(1.0, abs(G[u, i].item())))
    return worst


if __name__ == '__main__':
    results = []
    for T, Na, initial, mask in itertools.product((1, 2, 7, 13), (3, 2), (False, True), range(16)):
        results.append(check_case(T, 3, 6, Na, seed=1000 + T * 31 + Na * 7 + mask + 100 * initial, initial=initial, mask=mask))
    summ = {k_: max(r[k_] for r in results) for k_ in ('identity', 'chunk', 'delta_chain', 'dA_chain', 'dt_chain')}
    summ['cases'] = len(results)
    summ['max_no_kappa_when_terminal'] = max(r['no_kappa'] for r in results if r['mask'] & (1 | 2 | 8))
    summ['max_no_kappa_output_only'] = max(r['no_kappa'] for r in results if (r['mask'] & (1 | 2 | 8)) == 0)
    summ['max_kappa_output_only'] = max(r['kappa_norm'] for r in results if (r['mask'] & (1 | 2 | 8)) == 0)
    summ['max_G0_zero_init_output_only'] = max(r['G0'] for r in results if (r['mask'] & (1 | 2 | 8)) == 0 and not r['initial'])
    summ['min_G0_nonzero_init'] = min(r['G0'] for r in results if r['initial'] and r['T'] > 1)
    summ['unfolded_state_counterexample'] = unfolded_counterexample()
    summ['local_adjoint_errors'] = local_adjoint_check()
    summ['finite_difference'] = fd_check()
    print(json.dumps(summ, indent=1))
    json.dump(dict(summary=summ, cases=results), open(sys.argv[1] if len(sys.argv) > 1 else 'theory_check.json', 'w'), indent=1)
