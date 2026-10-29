"""Predeclared final analysis; refuses an incomplete or mismatched data set."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import *  # noqa: E402,F401,F403  (repository layout)
setup()
import sys,json,csv,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
METHODS=['bf16_sum','bf16_fp32_intermediates','bf16_boundary_legacy','bf16_boundary_policy','fp32_storage_tf32_dots','fp32_sum','local_fp32']
LABELS=['BF16 suffix','FP32 intermediates','Legacy boundary','Boundary policy','FP32 / TF32','FP32 / TF32x3','Local FP32 adjoint']
BASE=['bf16_sum','bf16_fp32_intermediates','fp32_storage_tf32_dots','fp32_sum','local_fp32']
resource={(s,m):json.loads((RESULTS/'resource_e2e'/f'{s}_{m}.json').read_text()) for s in range(4) for m in METHODS}
assert all(v['complete'] for v in resource.values())
def peak(c,m):return float(np.median(resource[c['shape_id'],m]['warmed_extra_peak_MB']))
cases=[];sha=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
for s in range(4):
    for seed in (1009,1013,1019):
        for slow in (0,1):
            c=json.loads((RESULTS/'final_study'/f'{s}_{seed}_{slow}.json').read_text())
            assert c['complete'] and c['protocol_sha256']==sha
            assert set(c['methods'])==set(METHODS)
            for m in c['methods'].values():
                assert 'error' not in m and all(m['replay_equal'].values())
            cases.append(c)
def err(c,m,k='Angles'):return c['methods'][m]['errors'][k]['rel_l2']
def cost(c,m):return c['methods'][m]['median_ms']
def qualifies(c,m,tol=.025):return err(c,m)<=tol and err(c,m,'DT')<=.01
def ci(vals,group):
    rng=np.random.default_rng(290131);vals=np.asarray(vals);group=np.asarray(group);unique=np.unique(group)
    med=[]
    for _ in range(10000):
        ids=np.concatenate([np.flatnonzero(group==g) for g in rng.choice(unique,len(unique),replace=True)])
        med.append(np.median(vals[ids]))
    return [float(x) for x in np.quantile(med,[.025,.975])]
summary={};rows=[]
for slow in (False,True):
    selected=[c for c in cases if c['slow_decay']==slow];groups=[c['shape_id'] for c in selected]
    error_ratio=[min(err(c,'bf16_sum'),err(c,'bf16_fp32_intermediates'))/err(c,'bf16_boundary_policy') for c in selected]
    overhead=[cost(c,'bf16_boundary_policy')/cost(c,'bf16_sum') for c in selected]
    fast=[];fastgroups=[];qualified=[]
    for c in selected:
        eligible=[m for m in BASE if qualifies(c,m)]
        win=min(eligible,key=lambda m:cost(c,m)) if eligible else None
        if qualifies(c,'bf16_boundary_policy'):
            assert win is not None
            fast.append(cost(c,win)/cost(c,'bf16_boundary_policy'));fastgroups.append(c['shape_id'])
        qualified.append(dict(shape=c['shape_id'],seed=c['seed'],policy_pass=qualifies(c,'bf16_boundary_policy'),fastest_qualified_nonboundary=win,ratio=None if win is None else cost(c,win)/cost(c,'bf16_boundary_policy')))
    methods={}
    for m in METHODS:
        methods[m]=dict(median_angle_error=float(np.median([err(c,m) for c in selected])),max_angle_error=float(max(err(c,m) for c in selected)),median_dt_error=float(np.median([err(c,m,'DT') for c in selected])),max_dt_error=float(max(err(c,m,'DT') for c in selected)),passes={str(tol):sum(qualifies(c,m,tol) for c in selected) for tol in (.01,.025,.05)},median_relative_time=float(np.median([cost(c,m)/cost(c,'bf16_sum') for c in selected])),median_relative_peak=float(np.median([peak(c,m)/peak(c,'bf16_sum') for c in selected])))
    study=dict(cases=len(selected),median_error_reduction=float(np.median(error_ratio)),error_reduction_cluster_ci95=ci(error_ratio,groups),max_error_regression=float(max(1/x for x in error_ratio)),median_overhead=float(np.median(overhead)),overhead_cluster_ci95=ci(overhead,groups),max_overhead=float(max(overhead)),policy_passes=sum(qualifies(c,'bf16_boundary_policy') for c in selected),median_qualified_speedup=float(np.median(fast)) if fast else None,qualified_speedup_cluster_ci95=ci(fast,fastgroups) if fast else None,qualified_cases=qualified,methods=methods)
    summary['stress' if slow else 'primary']=study
for c in cases:
    for m in METHODS:
        r=c['methods'][m]
        rows.append(dict(shape=c['shape_id'],seed=c['seed'],slow=c['slow_decay'],B=c['shape'][0],T=c['shape'][1],H=c['shape'][2],Hq=c['shape'][3],N=c['shape'][4],P=c['shape'][5],method=m,angle_error=err(c,m),dt_error=err(c,m,'DT'),angle_cosine=r['errors']['Angles']['cosine'],angle_sign_disagreement=r['errors']['Angles']['sign_disagreement'],forward_error=r['errors']['forward_rel_l2'],captured_backward_ms=cost(c,m),ordinary_backward_event_ms=float(np.median(r['ordinary_backward']['event_ms'])),ordinary_backward_wall_ms=float(np.median(r['ordinary_backward']['wall_ms'])),ordinary_forward_event_ms=float(np.median(r['ordinary_forward']['event_ms'])),promotion_event_ms=0 if r['promotion'] is None else float(np.median(r['promotion']['event_ms'])),corrected_shape_peak_MB=peak(c,m),cold_peak_diagnostic_MB=r['extra_peak_MB'],target_pass=qualifies(c,m)))
primary=summary['primary']
summary['criteria']=dict(error_reduction=primary['median_error_reduction']>=2,no_primary_regression=primary['max_error_regression']<=1.2,target_coverage=primary['policy_passes']>=10,qualifying_speed=primary['median_qualified_speedup'] is not None and primary['median_qualified_speedup']>=1.5,median_overhead=primary['median_overhead']<=1.25)
summary['all_primary_criteria_pass']=all(summary['criteria'].values());summary['protocol_sha256']=sha
(SUMMARIES/'final_study_summary.json').write_text(json.dumps(summary,indent=2))
with (SUMMARIES/'final_study_table.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220})
colors=['#687780','#a47b3a','#8f7eb4','#006a9c','#d24736','#648e27','#302a56'];markers=['o','s','^','D','v','P','X']
fig,axes=plt.subplots(1,2,figsize=(7.1,3.15),sharey=True)
for ax,slow in zip(axes,(False,True)):
    selected=[c for c in cases if c['slow_decay']==slow]
    for m,label,color,marker in zip(METHODS,LABELS,colors,markers):
        yy=[[err(c,m) for c in selected if c['shape_id']==s] for s in range(4)]
        ax.plot(range(4),[np.median(v) for v in yy],marker=marker,color=color,label=label,linewidth=1.1,markersize=4)
        for s,v in enumerate(yy):ax.scatter([s]*3,v,color=color,s=8,alpha=.35)
    ax.axhline(.025,color='#a00000',linestyle='--',linewidth=.8)
    ax.set_yscale('log');ax.set_xticks(range(4),['1023','2048','4097','8192']);ax.set_xlabel('Sequence length (shape also varies)');ax.set_title('Slow-decay stress' if slow else 'Primary cases');ax.grid(alpha=.15,which='major')
axes[0].set_ylabel('Angle-gradient relative L2 error')
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=7.2,bbox_to_anchor=(.5,-.025));fig.tight_layout(rect=[0,.13,1,1])
for ext in ('png','pdf','svg'):fig.savefig(FIGURES/f'final_gradient_errors.{ext}',bbox_inches='tight')
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(7.1,3.1));selected=[c for c in cases if not c['slow_decay']]
for m,label,color,marker in zip(METHODS,LABELS,colors,markers):
    for j,metric in enumerate(('time','memory')):
        vals=[]
        for s in range(4):
            cs=[c for c in selected if c['shape_id']==s]
            vals.append(np.median([cost(c,m)/cost(c,'bf16_sum') if metric=='time' else peak(c,m)/peak(c,'bf16_sum') for c in cs]))
        axes[j].plot(range(4),vals,marker=marker,color=color,label=label,linewidth=1.1,markersize=4)
for ax,title in zip(axes,('Captured backward time','Extra peak tensor memory')):
    ax.set_title(title);ax.axhline(1,color='gray',linewidth=.6);ax.set_yscale('log');ax.set_xticks(range(4),['1023','2048','4097','8192']);ax.set_xlabel('Sequence length (shape also varies)');ax.grid(alpha=.15)
axes[0].set_ylabel('Ratio to BF16 suffix')
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=7.2,bbox_to_anchor=(.5,-.025));fig.tight_layout(rect=[0,.13,1,1])
for ext in ('png','pdf','svg'):fig.savefig(FIGURES/f'final_cost_memory.{ext}',bbox_inches='tight')
plt.close(fig)
text=['# รอบ03: ผลทดสอบที่ล็อกไว้ล่วงหน้า','',f"ผลเกณฑ์หลัก: **{'PASS' if summary['all_primary_criteria_pass'] else 'FAIL'}** — นี่คือผลเกณฑ์ภายในที่ล็อกไว้ล่วงหน้า ไม่ใช่ผล peer review",'',
      'ข้อมูล: 24กรณี ×7วิธี, 12กรณีหลักและ12slow-decay stress; เมล็ดสุ่มใหม่3ค่า รูปทรง4ชุด RTX3080เครื่องเดียว สูตร/parameters ไม่เปลี่ยนหลังล็อก protocol ทุกกรณีตรวจ gradient ก่อนและหลัง CUDA graph replay แล้ว เก็บ raw timing, reference norm และ sign/cosine metrics ใน JSON/CSV','',
      f"Primary: median error reduction={primary['median_error_reduction']:.3f}×; policy ผ่าน target={primary['policy_passes']}/12; median matched-target speedup={primary['median_qualified_speedup']:.3f}×; median overhead เทียบ BF16 suffix={primary['median_overhead']:.3f}×; worst individual overhead={primary['max_overhead']:.3f}×",'',
      'เกณฑ์ target: angle relative L2≤0.025 และ DT≤0.01 เทียบกับ mathematical recurrence บน operands/cotangents ที่ quantize ตรงกัน; ขนาด error นี้ไม่ได้รับรองว่าทุกงานฝึกโมเดลยอมรับได้','']
for key,title in [('primary','กรณีหลัก'),('stress','กรณี decay ช้า')]:
    text+=['## '+title,'','| วิธี | median angle error | max angle error | pass1% /2.5% /5% | median time/BF16 | median peak/BF16 |','|---|---:|---:|---:|---:|---:|']
    for m,label in zip(METHODS,LABELS):
        v=summary[key]['methods'][m];p=v['passes'];text.append(f"| {label} | {v['median_angle_error']:.4g} | {v['max_angle_error']:.4g} | {p['0.01']} / {p['0.025']} / {p['0.05']} | {v['median_relative_time']:.3f} | {v['median_relative_peak']:.3f} |")
    text+=['']
text+=['## การตีความและข้อจำกัด','',
       '- ค่าเวลาในตารางคือ captured backward ครบทุก input gradient; ไม่ใช่ความเร็วฝึกโมเดลทั้งระบบ ordinary event/wall, forward และ precision promotion แยกอยู่ใน CSV/JSON',
       '- แก้ measurement memory: ค่า peak ครั้งแรกใน raw final_study รวม cold/autotune allocation จึงไม่ใช้เป็นหลักฐาน ตาราง/รูปนี้ใช้ resource_e2e ที่วัดหลัง warmup ใน process แยก จำนวน4shape×7วิธี seed1013 ค่าmemoryขึ้นกับshape ใช้ค่ารูปทรงเดียวกันในตารางstress ไม่ได้อ้างว่าวัดstressซ้ำ ค่าที่แก้คือ tensor allocation เพิ่มจาก caller inputs/cotangent สำหรับ forward+backward รวม precision promotion ไม่ใช่ memory reserved และไม่รวม CUDA graph pool',
       '- Kernel configurations ถูกเลือกจาก development; ไม่อ้างว่าเป็น implementation ที่เร็วที่สุดเท่าที่เป็นไปได้ และไม่สรุปข้ามฮาร์ดแวร์',
       '- Direct local-coordinate FP32 และ fullFP32 เป็นทางเลือกที่แม่นกว่ามาก ต้องคงไว้ในข้อเสนอแนะ ไม่ใช้ความเร็วของ BF16 ปิดบังความแม่นที่ต่างกัน',
       '- Confidence intervals ใน JSON bootstrap ตาม4shape clustersเท่านั้น จำนวนประชากรน้อย ไม่ถือ9timing repeatsเป็น9กรณีวิทยาศาสตร์อิสระ',
       '- ผลใหม่ไม่ลบผลลบของ iteration01/02: Lag oscillator ไม่ชนะ tuned factorization และ direct terminal correction ไม่ชนะ legacyอย่างสม่ำเสมอ',
       '- Mathematical boundary identity ไม่ได้ทำให้ทุก numerical implementation แม่นสม่ำเสมอทุกT: state/adjoint/phase errors และ conditioning ยังอยู่',
       '- ความใหม่ต้องวางบน explicit realization/qualified analysis/หลักฐานtrade-off พร้อมเครดิต Johnsonms และ Mamba-3 ไม่ใช่อ้างค้นพบ rotary equivalence หรือ reverse-mode differentiation',
       '', 'รูป: `final_gradient_errors.png` และ `final_cost_memory.png`; เวกเตอร์ PDF/SVG ชื่อเดียวกัน ข้อมูลหลัก: `final_study_summary.json`, `final_study_table.csv`, `final_study/*.json`','']
(NOTES/'ITERATION03_REPORT_TH.md').write_text('\n'.join(text),encoding='utf-8')
print(json.dumps({k:summary[k] for k in ('criteria','all_primary_criteria_pass')},indent=2))
