"""Repository layout. Every script imports this module so that paths do not depend on the working directory."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KERNELS = ROOT / 'kernels'          # modified mamba_ssm source trees (one per method family)
SRC = ROOT / 'src'                  # shared helpers: m3env, input generators, local-adjoint Triton kernels
REFERENCE = ROOT / 'reference'      # FP64 references and CPU correctness checks
EXPERIMENTS = ROOT / 'experiments'  # GPU studies and GPU validation
RESULTS = ROOT / 'results'          # raw and derived data
VALIDATION = RESULTS / 'validation'
SUMMARIES = RESULTS / 'summaries'
FIGURES = RESULTS / 'figures'
NOTES = ROOT / 'notes'
PROTOCOL = ROOT / 'protocols' / 'ITERATION03_FINAL_PROTOCOL.md'  # frozen; its SHA-256 is stored in every final_study result


def setup():
    """Make helper modules importable (m3env, m3_emulator, local_rotation_triton, manual_adjoint_reference, ...)."""
    for p in (SRC / 'm1017', SRC, REFERENCE):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    for d in (VALIDATION, SUMMARIES, FIGURES, NOTES):
        d.mkdir(parents=True, exist_ok=True)
