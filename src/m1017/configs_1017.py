"""Experiment grid for #1017 (shared by the GPU worker and the CPU analysis, so inputs can be regenerated)."""

VARIANTS_IO = [("D+Z+states", True, True, True), ("D+Z", True, True, False), ("D only", True, False, False),
               ("Z only", False, True, False), ("plain", False, False, False)]


def issue_grid(seeds=(0, 1, 2), hds=(64, 128)):
    """batch=4, seqlen=256, nheads=1, nheads_qk=1 — the configuration reported in the issue"""
    out = []
    for dtype in ("fp32", "bf16"):
        for hd in hds:
            for vname, D, Z, st in VARIANTS_IO:
                for seed in seeds:
                    out.append(dict(group="issue", name=f"{dtype}-hd{hd}-{vname}-s{seed}", batch=4, seqlen=256, nheads=1,
                                    nheads_qk=1, hd_qk=hd, hd_v=hd, dtype=dtype, has_D=D, has_Z=Z, has_states=st,
                                    seed=seed, bwd=(seed == 0 and hd == 64 and vname in ("D+Z+states", "plain"))))
    return out


def test_like():
    """shape family of tests/ops/triton/test_mamba3_siso.py::test_mamba3_siso_combined_batched (GQA, hd_qk 128, hd_v 64),
    reduced batch/heads so the float64 reference stays cheap"""
    return [dict(group="testlike", name=f"bf16-testlike-T2345-s{s}", batch=2, seqlen=2345, nheads=8, nheads_qk=2,
                 hd_qk=128, hd_v=64, dtype="bf16", has_D=True, has_Z=True, has_states=True, seed=40 + s, bwd=False)
            for s in range(2)]


def long_seq(Ts=(256, 1024, 4096, 16384)):
    return [dict(group="long", name=f"fp32-hd64-D+Z-T{T}", batch=1, seqlen=T, nheads=4, nheads_qk=1, hd_qk=64, hd_v=64,
                 dtype="fp32", has_D=True, has_Z=True, has_states=False, seed=7, bwd=False) for T in Ts]


def tsweep(Ts=(256, 512, 1024, 2048, 4096), seeds=(11, 12, 13)):
    """P1: dAngles error vs seqlen (bwd only matters). B=1, H=4; the float64 reference runs one head at a time.
    'plain' = no input states (the loss still sees the final states)."""
    return [dict(group="tsweep", name=f"bf16-hd64-{vname}-T{T}-s{seed}", batch=1, seqlen=T, nheads=4, nheads_qk=1,
                 hd_qk=64, hd_v=64, dtype="bf16", has_D=D, has_Z=Z, has_states=st, seed=seed, bwd=True)
            for T in Ts for vname, D, Z, st in (VARIANTS_IO[0], VARIANTS_IO[4]) for seed in seeds]


GROUPS = {"issue": issue_grid, "testlike": test_like, "long": long_seq, "tsweep": tsweep}


def group_configs(groups):
    return [c for g in groups for c in GROUPS[g]()]


def all_configs(quick=False):
    if quick:
        return issue_grid(seeds=(0,), hds=(64,)) + long_seq(Ts=(256, 4096))
    return issue_grid() + test_like() + long_seq()
