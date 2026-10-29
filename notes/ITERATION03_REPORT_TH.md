# รอบ03: ผลทดสอบที่ล็อกไว้ล่วงหน้า

ผลเกณฑ์หลัก: **PASS** — นี่คือผลเกณฑ์ภายในที่ล็อกไว้ล่วงหน้า ไม่ใช่ผล peer review

ข้อมูล: 24กรณี ×7วิธี, 12กรณีหลักและ12slow-decay stress; เมล็ดสุ่มใหม่3ค่า รูปทรง4ชุด RTX3080เครื่องเดียว สูตร/parameters ไม่เปลี่ยนหลังล็อก protocol ทุกกรณีตรวจ gradient ก่อนและหลัง CUDA graph replay แล้ว เก็บ raw timing, reference norm และ sign/cosine metrics ใน JSON/CSV

Primary: median error reduction=5.131×; policy ผ่าน target=12/12; median matched-target speedup=1.648×; median overhead เทียบ BF16 suffix=1.042×; worst individual overhead=1.305×

เกณฑ์ target: angle relative L2≤0.025 และ DT≤0.01 เทียบกับ mathematical recurrence บน operands/cotangents ที่ quantize ตรงกัน; ขนาด error นี้ไม่ได้รับรองว่าทุกงานฝึกโมเดลยอมรับได้

## กรณีหลัก

| วิธี | median angle error | max angle error | pass1% /2.5% /5% | median time/BF16 | median peak/BF16 |
|---|---:|---:|---:|---:|---:|
| BF16 suffix | 0.09937 | 0.1718 | 0 / 0 / 0 | 1.000 | 1.000 |
| FP32 intermediates | 0.08235 | 0.1483 | 0 / 0 / 0 | 1.172 | 1.234 |
| Legacy boundary | 0.02437 | 0.02862 | 0 / 7 / 12 | 1.035 | 1.000 |
| Boundary policy | 0.01604 | 0.01679 | 0 / 12 / 12 | 1.042 | 1.001 |
| FP32 / TF32 | 0.008233 | 0.01376 | 9 / 12 / 12 | 1.686 | 2.511 |
| FP32 / TF32x3 | 8.639e-06 | 1.25e-05 | 12 / 12 / 12 | 19.001 | 2.375 |
| Local FP32 adjoint | 2.736e-07 | 2.763e-07 | 12 / 12 / 12 | 11.432 | 12.108 |

## กรณี decay ช้า

| วิธี | median angle error | max angle error | pass1% /2.5% /5% | median time/BF16 | median peak/BF16 |
|---|---:|---:|---:|---:|---:|
| BF16 suffix | 0.01489 | 0.0219 | 3 / 6 / 6 | 1.000 | 1.000 |
| FP32 intermediates | 0.0135 | 0.0195 | 3 / 8 / 8 | 1.087 | 1.234 |
| Legacy boundary | 0.00585 | 0.006231 | 12 / 12 / 12 | 1.045 | 1.000 |
| Boundary policy | 0.005675 | 0.00601 | 12 / 12 / 12 | 1.042 | 1.001 |
| FP32 / TF32 | 0.002156 | 0.002993 | 12 / 12 / 12 | 1.753 | 2.511 |
| FP32 / TF32x3 | 8.79e-06 | 8.962e-06 | 12 / 12 / 12 | 18.335 | 2.375 |
| Local FP32 adjoint | 5.802e-06 | 6.405e-06 | 12 / 12 / 12 | 10.344 | 12.108 |

## การตีความและข้อจำกัด

- ค่าเวลาในตารางคือ captured backward ครบทุก input gradient; ไม่ใช่ความเร็วฝึกโมเดลทั้งระบบ ordinary event/wall, forward และ precision promotion แยกอยู่ใน CSV/JSON
- แก้ measurement memory: ค่า peak ครั้งแรกใน raw final_study รวม cold/autotune allocation จึงไม่ใช้เป็นหลักฐาน ตาราง/รูปนี้ใช้ resource_e2e ที่วัดหลัง warmup ใน process แยก จำนวน4shape×7วิธี seed1013 ค่าmemoryขึ้นกับshape ใช้ค่ารูปทรงเดียวกันในตารางstress ไม่ได้อ้างว่าวัดstressซ้ำ ค่าที่แก้คือ tensor allocation เพิ่มจาก caller inputs/cotangent สำหรับ forward+backward รวม precision promotion ไม่ใช่ memory reserved และไม่รวม CUDA graph pool
- Kernel configurations ถูกเลือกจาก development; ไม่อ้างว่าเป็น implementation ที่เร็วที่สุดเท่าที่เป็นไปได้ และไม่สรุปข้ามฮาร์ดแวร์
- Direct local-coordinate FP32 และ fullFP32 เป็นทางเลือกที่แม่นกว่ามาก ต้องคงไว้ในข้อเสนอแนะ ไม่ใช้ความเร็วของ BF16 ปิดบังความแม่นที่ต่างกัน
- Confidence intervals ใน JSON bootstrap ตาม4shape clustersเท่านั้น จำนวนประชากรน้อย ไม่ถือ9timing repeatsเป็น9กรณีวิทยาศาสตร์อิสระ
- ผลใหม่ไม่ลบผลลบของ iteration01/02: Lag oscillator ไม่ชนะ tuned factorization และ direct terminal correction ไม่ชนะ legacyอย่างสม่ำเสมอ
- Mathematical boundary identity ไม่ได้ทำให้ทุก numerical implementation แม่นสม่ำเสมอทุกT: state/adjoint/phase errors และ conditioning ยังอยู่
- ความใหม่ต้องวางบน explicit realization/qualified analysis/หลักฐานtrade-off พร้อมเครดิต Johnsonms และ Mamba-3 ไม่ใช่อ้างค้นพบ rotary equivalence หรือ reverse-mode differentiation

รูป: `final_gradient_errors.png` และ `final_cost_memory.png`; เวกเตอร์ PDF/SVG ชื่อเดียวกัน ข้อมูลหลัก: `final_study_summary.json`, `final_study_table.csv`, `final_study/*.json`
