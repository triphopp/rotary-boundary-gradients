"""gpu_worker_1017.py — run one source variant of mamba3_siso_combined on the GPU and score it (issue #1017).

    python gpu_worker_1017.py --variant orig [--quick] [--no-bwd]
    python gpu_worker_1017.py --variant anglefp32 --group tsweep     # P1: dAngles vs seqlen -> summary_tsweep.json

* inputs: m3_emulator.make_inputs (numpy, seeded)  -> identical across variants, regenerable on any machine
* forward : kernel vs float64 reference (m3_emulator.reference, validated against an O(T) recurrence),
            kernel vs the CPU emulator of the *same* rounding policy (how much is left unexplained),
            kernel vs the repo's own fp32 reference mamba3_siso_fwd_ref (= the methodology of the issue)
* backward (selected configs): kernel grads vs float64 autograd of ref_torch64.mamba3_siso_ref64
* angle drift: angle_dt_fwd alone on long sequences vs float64 cumsum
* dtheta breakdown (every bwd config): the per-token dL/dtheta_t that compute_dqktheta hands to angle_dt_bwd, and
  their reverse cumsum summed exactly in float64, each vs the float64 reference (separates bad terms from bad summation)
* oracle_* variants: the float64 reference supplies the gradient carried across chunk boundaries in angle_dt_bwd
Writes results_gpu/<variant>/summary.json and outputs.npz (with --group G: summary_G.json only, no drift).
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time
import traceback
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]

import numpy as np  # noqa: E402
import m3env  # noqa: E402
import m3_emulator as E  # noqa: E402
import configs_1017 as C  # noqa: E402

EMU_FLAGS = {
    "orig": E.ALL_FLAGS, "orig_autotune": E.ALL_FLAGS, "exacttrig": E.ALL_FLAGS,
    "anglefp32": tuple(f for f in E.ALL_FLAGS if f != "ang_store"),
    "anglefp32_exacttrig": tuple(f for f in E.ALL_FLAGS if f != "ang_store"),
}
GRAD_NAMES = ["Q", "K", "V", "ADT", "DT", "Trap", "Q_bias", "K_bias", "Angles", "D", "Z",
              "In_Angle", "In_SSM", "In_K", "In_V"]


def to_torch(inp, dtype, dev, torch, requires_grad=False):
    lp = torch.float32 if dtype == "fp32" else torch.bfloat16

    def T(x, dt):
        if x is None:
            return None
        t = torch.from_numpy(np.ascontiguousarray(x)).to(dev).to(dt)
        return t.requires_grad_(requires_grad)

    st = inp["Input_States"]
    return dict(Q=T(inp["Q"], lp), K=T(inp["K"], lp), V=T(inp["V"], lp), ADT=T(inp["ADT"], torch.float32),
                DT=T(inp["DT"], torch.float32), Trap=T(inp["Trap"], lp), Q_bias=T(inp["Q_bias"], lp),
                K_bias=T(inp["K_bias"], lp), Angles=T(inp["Angles"], torch.float32), D=T(inp["D"], torch.float32),
                Z=T(inp["Z"], lp),
                Input_States=None if st is None else tuple(T(x, torch.float32) for x in st))


def load_test_ref(src):
    """import mamba3_siso_fwd_ref from the snapshot's test file (stub pytest if missing)"""
    try:
        import pytest  # noqa: F401
    except ImportError:
        class _Mark:
            def __getattr__(self, name):
                return lambda *a, **k: (lambda f: f)
        stub = types.ModuleType("pytest")
        stub.mark = _Mark()
        sys.modules["pytest"] = stub
    path = os.path.join(src, "tests", "ops", "triton", "test_mamba3_siso.py")
    spec = importlib.util.spec_from_file_location("m3_tests_snapshot", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.mamba3_siso_fwd_ref


def circ(a, b):
    d = np.remainder(np.asarray(a, np.float64) - np.asarray(b, np.float64) + math.pi, 2 * math.pi) - math.pi
    return d


def gstats(ker, ref):
    ker = np.asarray(ker, np.float64)
    ref = np.asarray(ref, np.float64)
    return dict(p95=E.p95_rel(ker, ref), norm=E.normwise(ker, ref), maxabs=float(np.max(np.abs(ker - ref))))


def dtheta_breakdown(cap, dth64, R64, cot_ang, dang_k, dang_r):
    """Split the dAngles error into (a) the per-token terms dL/dtheta_t and (b) their reverse cumsum.
    cap['g'] (B,T,H,da) = kernel terms without the final-angle-state cotangent, which the kernel adds as
    grad_output_state = cap['gos'] (B,H,da); the reference's dth64[..., -1, :] contains that cotangent.
    revcumsum_exact sums the kernel's terms in float64: if it is as wrong as dAngles, the summation is not the problem.
    zerosum_*: rms over (B,H,da) of sum_t dL/dtheta_t (0 for the exact gradient when there are no input states)."""
    import torch
    g_k = cap["g"].permute(0, 2, 1, 3).double()
    gos = cap["gos"].double() if cap.get("gos") is not None else torch.zeros_like(g_k[:, :, 0])
    g_r = dth64.clone()
    g_r[:, :, -1] -= torch.as_tensor(cot_ang, device=g_r.device)
    R_k = g_k.flip(2).cumsum(2).flip(2) + gos[:, :, None]

    def rms(x):
        return float(x.double().pow(2).mean().sqrt())

    def st(k, r):
        k, r = k.double(), r.double()
        return dict(norm=float((k - r).norm() / r.norm()), rmsabs=rms(k - r), maxabs=float((k - r).abs().max()),
                    ref_rms=rms(r))
    T = R64.shape[2]
    at = sorted({max(0, int(T * f) - 1) for f in (0.0, 0.25, 0.5, 0.75, 1.0)})
    return dict(term=st(g_k, g_r), revcumsum_exact=st(R_k, R64), dAngles=st(dang_k, dang_r),
                zerosum_ref_rms=rms(R64[:, :, 0]), zerosum_kernel_rms=rms(R_k[:, :, 0]),
                revcumsum_err_rms_at=[rms((R_k - R64)[:, :, i]) for i in at], at_index=at)


def carry_split(c, r):
    """Fix B carry (B,H,nch-1,da) vs exact. The error is split into the part that is the same at every boundary
    (mean over chunks: what an error in const_i does) and the part that changes from boundary to boundary
    (the error of <dS_c, S_c J>). last_rms: error at the last boundary, where the carry is the local sum only."""
    d = c - r
    off = d.mean(2, keepdim=True)

    def rms(x):
        return float(x.pow(2).mean().sqrt())
    return dict(norm=float(d.norm() / r.norm()), rmsabs=rms(d), ref_rms=rms(r), offset_rms=rms(off.expand_as(d)),
                resid_rms=rms(d - off), last_rms=rms(d[:, :, -1]))


def ref_backward(cpath, inp, cfg, cot, ref_out, ref64, dev, torch):
    """float64 autograd of ref_torch64 -> (forward check, dL/dtheta_t (B,H,T,da) on dev, {grad name: np array}).
    Cached in ref_cache (identical for every variant). With one Q/K head the heads are independent, so the reference
    runs one head at a time (memory O(T^2) per head, T=4096 fits on a 10 GB card); Q/K grads add up over heads."""
    if os.path.exists(cpath):
        z = np.load(cpath)
        return float(z["chk"]), torch.from_numpy(z["dth"]).to(dev), {k[2:]: z[k] for k in z.files if k.startswith("g_")}
    t64 = {k: (None if v is None else (tuple(x.double().requires_grad_() for x in v) if isinstance(v, tuple)
                                        else v.double().requires_grad_())) for k, v in to_torch(inp, cfg["dtype"], dev, torch).items()}
    H = cfg["nheads"]
    heads = [slice(h, h + 1) for h in range(H)] if cfg["nheads_qk"] == 1 else [slice(0, H)]
    outs, dths = [], []
    for hs in heads:
        hd = lambda x, dim: None if x is None else x[(slice(None),) * dim + (hs,)]  # noqa: E731
        st = t64["Input_States"]
        keep = {}
        r64 = ref64(t64["Q"], t64["K"], hd(t64["V"], 2), hd(t64["ADT"], 1), hd(t64["DT"], 1), hd(t64["Trap"], 1),
                    hd(t64["Q_bias"], 0), hd(t64["K_bias"], 0), hd(t64["Angles"], 2), hd(t64["D"], 0), hd(t64["Z"], 2),
                    None if st is None else tuple(hd(x, 1) for x in st), keep=keep)
        cot_h = [c[:, :, hs] if i == 0 else c[:, hs] for i, c in enumerate(cot)]
        torch.autograd.backward(list(r64), [torch.from_numpy(np.ascontiguousarray(c)).to(dev) for c in cot_h])
        outs.append(r64[0].detach())
        dths.append(keep["theta"].grad.detach())
        del r64, keep
    chk = E.normwise(torch.cat(outs, 2).cpu().numpy(), ref_out)
    dth = torch.cat(dths, 1)
    leaves = [t64[n] for n in GRAD_NAMES[:11]] + (list(t64["Input_States"]) if t64["Input_States"] else [None] * 4)
    grads = {n: x.grad.cpu().numpy() for n, x in zip(GRAD_NAMES, leaves) if x is not None and x.grad is not None}
    np.savez_compressed(cpath, chk=chk, dth=dth.cpu().numpy(), **{"g_" + k: v for k, v in grads.items()})
    return chk, dth, grads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-bwd", action="store_true")
    ap.add_argument("--no-drift", action="store_true")
    ap.add_argument("--group", help="comma list of config groups (issue,testlike,long,tsweep); writes summary_<group>.json")
    a = ap.parse_args()

    import make_variants
    src = os.path.join(make_variants.OUT, a.variant)
    if not os.path.isdir(src):
        make_variants.build([a.variant])
    m3env.use_source(src)
    import torch
    from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined
    from mamba_ssm.ops.triton.mamba3.angle_dt import angle_dt_fwd
    import mamba_ssm.ops.triton.mamba3.angle_dt as angle_dt_mod
    import mamba_ssm.ops.triton.mamba3.mamba3_siso_combined as combined_mod
    oracle = getattr(angle_dt_mod, "ORACLE", None)  # only the oracle_* / bnd_state variants have it
    from_states = getattr(combined_mod, "BOUNDARY_FROM_STATES", False)  # bnd_state variants: Fix B prototype
    cap = {}
    _angle_dt_bwd = combined_mod.angle_dt_bwd

    def angle_dt_bwd_spy(*args, **kw):  # records dL/dtheta_t (kernel) as handed to the reverse cumsum
        cap["g"] = (kw["grad_out"] if "grad_out" in kw else args[0]).detach().float().clone()
        gos = kw.get("grad_output_state")
        cap["gos"] = None if gos is None else gos.detach().float().clone()
        return _angle_dt_bwd(*args, **kw)
    combined_mod.angle_dt_bwd = angle_dt_bwd_spy
    from ref_torch64 import mamba3_siso_ref64
    test_ref = load_test_ref(src)

    dev = "cuda"
    outdir = os.path.join(HERE, "results_gpu", a.variant)
    os.makedirs(outdir, exist_ok=True)
    cache = os.path.join(HERE, "ref_cache")
    os.makedirs(cache, exist_ok=True)
    summary = dict(variant=a.variant, env=m3env.environment(), started=time.strftime("%Y-%m-%d %H:%M:%S"),
                   forward={}, backward={}, drift={}, errors={})
    arrays = {}
    fp32_only = a.variant.startswith("fp32_")
    groups = a.group.split(",") if a.group else None
    if groups:
        a.no_drift = True
    t0 = time.time()

    for cfg in (C.group_configs(groups) if groups else C.all_configs(a.quick)):
        name = cfg["name"]
        if fp32_only and cfg["dtype"] != "fp32":
            continue
        inp = E.make_inputs(cfg["seed"], cfg["batch"], cfg["seqlen"], cfg["nheads"], cfg["nheads_qk"], cfg["hd_qk"],
                            cfg["hd_v"], cfg["dtype"], cfg["has_D"], cfg["has_Z"], cfg["has_states"])
        cpath = os.path.join(cache, name + ".npz")
        if os.path.exists(cpath):
            z = np.load(cpath)
            ref = (z["out"], z["ang"], z["ssm"], z["k"], z["v"])
        else:
            o, (fa, fs, fk, fv) = E.reference(inp)
            ref = (o, fa, fs, fk, fv)
            np.savez_compressed(cpath, out=o, ang=fa, ssm=fs, k=fk, v=fv)
        try:
            t = to_torch(inp, cfg["dtype"], dev, torch)
            with torch.no_grad():
                torch.cuda.synchronize()
                tk = time.time()
                res = mamba3_siso_combined(t["Q"], t["K"], t["V"], t["ADT"], t["DT"], t["Trap"], t["Q_bias"],
                                           t["K_bias"], t["Angles"], t["D"], t["Z"], t["Input_States"],
                                           chunk_size=64, return_final_states=True)
                torch.cuda.synchronize()
                tk = time.time() - tk
            out, fa, fs, fk, fv = [r.float().cpu().numpy().astype(np.float64) for r in res]
            rec = dict(cfg=cfg, seconds_first_call=tk,
                       out=E.stats(out, ref[0]),
                       final_angle_maxabs_rad=float(np.max(np.abs(circ(fa, ref[1])))),
                       final_ssm=gstats(fs, ref[2]), final_k=gstats(fk, ref[3]), final_v=gstats(fv, ref[4]),
                       dtypes=dict(out=str(res[0].dtype), angle_state=str(res[1].dtype)))
            fl = EMU_FLAGS.get(a.variant)
            if fl is not None:
                emu, _ = E.forward(inp, fl)
                rec["unexplained_by_emulator"] = E.normwise(out, emu) / 1.0
                rec["unexplained_rel_to_error"] = E.normwise(out, emu) * np.linalg.norm(emu) / max(
                    1e-300, np.linalg.norm(out - ref[0]))
            if cfg["group"] == "issue" and cfg["seed"] == 0:
                with torch.no_grad():
                    r32, _ = test_ref(t["Q"], t["K"], t["V"], t["ADT"], t["DT"], t["Trap"], t["Q_bias"], t["K_bias"],
                                      t["Angles"], t["D"], t["Z"], t["Input_States"])
                r32 = r32.float().cpu().numpy().astype(np.float64)
                rec["vs_repo_fp32_ref"] = E.stats(out, r32)
                rec["repo_fp32_ref_vs_f64"] = E.stats(r32, ref[0])
            summary["forward"][name] = rec
            if cfg["group"] != "long":
                arrays[name] = out.astype(np.float32)
            print(f"[{a.variant}] {name:38s} p95 {rec['out']['p95']:.2e} norm {rec['out']['norm']:.2e}"
                  + (f"  unexplained {rec['unexplained_by_emulator']:.1e}" if "unexplained_by_emulator" in rec else "")
                  + (f"  repo-ref p95 {rec['vs_repo_fp32_ref']['p95']:.2e}" if "vs_repo_fp32_ref" in rec else ""),
                  flush=True)
        except Exception as e:
            summary["errors"][name] = traceback.format_exc()
            print(f"[{a.variant}] {name}: ERROR {repr(e)[:300]}", flush=True)
            continue

        if cfg["bwd"] and not a.no_bwd and not fp32_only:  # bwd kernels require the bf16 operands (fp32_* = fwd-only probes)
            try:
                rng = np.random.default_rng(1000 + cfg["seed"])
                cot = [rng.standard_normal(x.shape) for x in ref]
                chk, dth64, g64 = ref_backward(os.path.join(cache, name + "_bwd64.npz"), inp, cfg, cot, ref[0],
                                               mamba3_siso_ref64, dev, torch)
                R64 = dth64.flip(2).cumsum(2).flip(2)     # sum_{t'>=t} dL/dtheta_t'
                CH = 64
                nch = -(-cfg["seqlen"] // CH)
                bnd = torch.zeros(dth64.shape[:2] + (nch, dth64.shape[-1]), dtype=torch.float64, device=dev)
                if nch > 1:
                    bnd[:, :, :-1] = R64[:, :, CH::CH][:, :, :nch - 1]  # exact carry into chunk c
                if oracle is not None:
                    if not from_states:  # bnd_state variants compute the carry themselves (from S_c, dS_c)
                        oracle["boundary"] = bnd.float()
                    oracle["used"] = 0
                cap.clear()
                tk = to_torch(inp, cfg["dtype"], dev, torch, requires_grad=True)
                res = mamba3_siso_combined(tk["Q"], tk["K"], tk["V"], tk["ADT"], tk["DT"], tk["Trap"], tk["Q_bias"],
                                           tk["K_bias"], tk["Angles"], tk["D"], tk["Z"], tk["Input_States"],
                                           chunk_size=64, return_final_states=True)
                torch.autograd.backward(list(res), [torch.from_numpy(c).to(dev).to(r.dtype) for c, r in zip(cot, res)])
                carry_rec = None
                if oracle is not None:
                    assert oracle.pop("used") == 1, "boundary carry was not consumed by angle_dt_bwd"
                    if from_states:
                        c = oracle.pop("last_from_states").double()[:, :, :nch - 1]
                        carry_rec = carry_split(c, bnd[:, :, :nch - 1]) if nch > 1 else None
                    else:
                        oracle.pop("boundary")
                gk = [tk[n] for n in GRAD_NAMES[:11]] + (list(tk["Input_States"]) if tk["Input_States"] else [None] * 4)
                brec = dict(ref64_forward_check=chk)
                for n, x in zip(GRAD_NAMES, gk):
                    if x is None or x.grad is None or n not in g64:
                        continue
                    brec[n] = gstats(x.grad.float().cpu().numpy(), g64[n])
                brec["dtheta"] = dtheta_breakdown(cap, dth64, R64, cot[1], tk["Angles"].grad,
                                                  torch.from_numpy(g64["Angles"]).to(dev))
                if carry_rec is not None:
                    brec["dtheta"]["carry_from_states"] = carry_rec
                summary["backward"][name] = brec
                print(f"[{a.variant}] bwd {name}: " + "  ".join(f"d{n} {v['norm']:.1e}" for n, v in brec.items()
                                                                 if isinstance(v, dict) and "norm" in v), flush=True)
                b = brec["dtheta"]
                print(f"[{a.variant}]   dtheta: term norm {b['term']['norm']:.2e} | revcumsum of kernel terms (f64) norm "
                      f"{b['revcumsum_exact']['norm']:.2e} rmsabs {b['revcumsum_exact']['rmsabs']:.2e} "
                      f"(ref rms {b['revcumsum_exact']['ref_rms']:.2e}) | sum_t ref {b['zerosum_ref_rms']:.1e} "
                      f"ker {b['zerosum_kernel_rms']:.1e} | dAngles norm {b['dAngles']['norm']:.2e} "
                      f"rmsabs {b['dAngles']['rmsabs']:.2e}"
                      + (f" | carry from states vs exact: norm {carry_rec['norm']:.2e} (const-offset rms "
                         f"{carry_rec['offset_rms']:.1e}, per-boundary rms {carry_rec['resid_rms']:.1e}, last-boundary rms "
                         f"{carry_rec['last_rms']:.1e}, ref rms {carry_rec['ref_rms']:.1e})" if carry_rec else ""), flush=True)
            except Exception as e:
                summary["errors"]["bwd " + name] = traceback.format_exc()
                print(f"[{a.variant}] bwd {name}: ERROR {repr(e)[:300]}", flush=True)

    if not a.no_drift:
        # phase accumulation in angle_dt_fwd alone (inputs bf16 as mamba3_siso_combined passes them)
        for T in ((256, 4096) if a.quick else (256, 4096, 65536, 262144)):
            try:
                rng = np.random.default_rng(5)
                B, H, da = 2, 4, 16
                ang = E.bf16(rng.standard_normal((B, T, H, da)))
                if fp32_only:
                    ang = E.f32(ang)
                dt = E.f32(np.exp(rng.uniform(0, 1, (B, H, T)) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)))
                inc = np.tanh(ang).transpose(0, 2, 1, 3) * math.pi * dt[..., None]
                ref_th = np.mod(np.cumsum(inc, axis=2), 2 * math.pi)  # (B,H,T,da)
                a_t = torch.from_numpy(ang).to(dev).to(torch.float32 if fp32_only else torch.bfloat16)
                d_t = torch.from_numpy(dt).to(dev).to(torch.float32)
                with torch.no_grad():
                    th, st = angle_dt_fwd(a_t, d_t, chunk_size=64, return_output_state=True)
                th_dtype = str(th.dtype)
                th = th.float().cpu().numpy().astype(np.float64).transpose(0, 2, 1, 3)
                st = st.float().cpu().numpy().astype(np.float64)
                e_all = circ(th, ref_th)
                e_fin = circ(st, ref_th[:, :, -1, :])
                idx = [int(T * f) - 1 for f in (0.125, 0.25, 0.5, 1.0)]
                summary["drift"][str(T)] = dict(
                    dtype_theta=th_dtype, max_abs_all=float(np.max(np.abs(e_all))),
                    rms_all=float(np.sqrt(np.mean(e_all ** 2))), final_state_max=float(np.max(np.abs(e_fin))),
                    mean_signed_at=[float(np.mean(e_all[:, :, i, :])) for i in idx],
                    rms_at=[float(np.sqrt(np.mean(e_all[:, :, i, :] ** 2))) for i in idx], at_index=idx)
                print(f"[{a.variant}] drift T={T}: max {summary['drift'][str(T)]['max_abs_all']:.2e} "
                      f"rms {summary['drift'][str(T)]['rms_all']:.2e} rms@end {summary['drift'][str(T)]['rms_at'][-1]:.2e}"
                      f" mean@end {summary['drift'][str(T)]['mean_signed_at'][-1]:+.2e}", flush=True)
            except Exception as e:
                summary["errors"][f"drift {T}"] = traceback.format_exc()
                print(f"[{a.variant}] drift {T}: ERROR {repr(e)[:300]}", flush=True)

    summary["seconds_total"] = time.time() - t0
    if groups:
        summary["groups"] = groups
        json.dump(summary, open(os.path.join(outdir, f"summary_{'_'.join(groups)}.json"), "w"), indent=1)
    else:
        json.dump(summary, open(os.path.join(outdir, "summary.json"), "w"), indent=1)
        np.savez_compressed(os.path.join(outdir, "outputs.npz"), **arrays)
    print(f"[{a.variant}] done in {summary['seconds_total']:.0f}s, errors: {len(summary['errors'])}")


if __name__ == "__main__":
    main()
