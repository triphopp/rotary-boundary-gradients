import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import *  # noqa: E402,F401,F403  (repository layout)
setup()
import json,shutil,hashlib
from pathlib import Path
import statistics as st
s=json.loads((SUMMARIES/'final_study_summary.json').read_text());p=s['primary']
assert s['all_primary_criteria_pass']
macros={
 'ErrorGain':f"{p['median_error_reduction']:.2f}$\\times$",'SpeedGain':f"{p['median_qualified_speedup']:.2f}$\\times$",
 'ErrorCI':f"{p['error_reduction_cluster_ci95'][0]:.2f}--{p['error_reduction_cluster_ci95'][1]:.2f}",
 'SpeedCI':f"{p['qualified_speedup_cluster_ci95'][0]:.2f}--{p['qualified_speedup_cluster_ci95'][1]:.2f}",
 'Overhead':f"{p['median_overhead']:.3f}$\\times$",'WorstOverhead':f"{p['max_overhead']:.3f}$\\times$"}
methods=['bf16_sum','bf16_fp32_intermediates','bf16_boundary_legacy','bf16_boundary_policy','fp32_storage_tf32_dots','fp32_sum','local_fp32']
names=['BF16 suffix','FP32 intermediates','Legacy boundary','Boundary policy','FP32 / TF32','FP32 / TF32x3','Local FP32 adjoint']
rows=[]
for m,n in zip(methods,names):
    v=p['methods'][m];rows.append(f"{n}&{v['median_angle_error']:.3g}&{v['passes']['0.025']}&{v['median_relative_time']:.2f}\\\\")
macros['MainTableRows']='\n'.join(rows)
resources={(i,m):json.loads((RESULTS/'resource_e2e'/f'{i}_{m}.json').read_text()) for i in range(4) for m in methods}
ratios=[resources[i,'fp32_storage_tf32_dots']['median_captured_e2e_ms']/resources[i,'bf16_boundary_policy']['median_captured_e2e_ms'] for i in range(4)]
peaks=[st.median(resources[i,'bf16_boundary_policy']['warmed_extra_peak_MB']) for i in range(4)]
memory_ratios=[peaks[i]/st.median(resources[i,'bf16_sum']['warmed_extra_peak_MB']) for i in range(4)]
macros['ResourceParagraph']=(f"A separate warmed, fresh-process check over the four shapes includes forward, all backward work, precision promotion and conversion to caller gradient dtypes. Policy/ordinary extra peak tensor allocation ratios are {min(memory_ratios):.4f}--{max(memory_ratios):.4f}, with policy peaks {min(peaks):.2f}--{max(peaks):.2f} MiB. Complete-primitive captured speedups over TF32 range {min(ratios):.2f}--{max(ratios):.2f}$\\times$. These supplementary single-process measurements are not full-model training throughput. The initial cold/autotune peak field is archived but excluded from memory conclusions.")
(SUMMARIES/'analysis_macros.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n')
table=['# Warmed memory and complete-primitive measurements','','Four existing primary shapes, seed1013, seven methods; each measurement process is independent. Full forward/backward includes dtype promotion and caller-gradient conversion. No model projection, optimizer or training iteration is implied.','','| Shape | Method | Extra peak MiB | Captured full primitive ms | Ordinary wall ms |','|---|---|---:|---:|---:|']
for i in range(4):
    for m,n in zip(methods,names):
        r=resources[i,m];table.append(f"| {i} | {n} | {st.median(r['warmed_extra_peak_MB']):.3f} | {r['median_captured_e2e_ms']:.4f} | {st.median(r['ordinary_e2e_wall_ms']):.4f} |")
(NOTES/'RESOURCE_E2E_REPORT.md').write_text('\n'.join(table)+'\n')
print('Wrote results/summaries/analysis_macros.tex and notes/RESOURCE_E2E_REPORT.md; complete primitive speedups',ratios,'memory ratios',memory_ratios)
