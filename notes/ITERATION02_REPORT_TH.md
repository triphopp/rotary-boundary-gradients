# รอบ 02: ตรวจสูตร boundary-gradient และแก้ terminal correction

วันที่ 28 ตุลาคม 2026 · **สถานะ: boundary formulation มีประโยชน์จริงในขอบเขตที่ทดสอบ แต่ direct terminal correction ไม่ผ่านเกณฑ์ว่าจะเพิ่มประโยชน์เหนือสูตรเดิมอย่างเพียงพอในชุดยืนยันนี้ งานยังไม่พร้อมส่ง**

## สิ่งที่ทำเสร็จ

- พิสูจน์สูตรรวม final SSM/K/angle cotangents และแก้ความหมายของ boundary state ให้รวม trapezoidal previous-input term (`BOUNDARY_THEORY.md`)
- ตรวจ 256 cases / 4,800 cuts ด้วย sequential autograd เทียบ attention reference และ finite differences
- สร้าง reference อีกแบบที่เขียน reverse adjoint เองด้วย NumPy FP64 แล้วตรวจเทียบ autograd 36 cases คลาดสูงสุด 1.56e-15
- สร้าง GPU prototype แยกจาก branch เดิม ทดสอบ 12 development cases เดิมพบ direct terminal ลด error ทุกกรณี แต่ไม่ใช้เป็นหลักฐาน held out
- ตรวจ source ใกล้ที่สุดที่เผยแพร่ก่อน: phanerozoic/mamba3 ยังใช้ reverse cumsum ใน wrapper ที่ตรวจ ให้เครดิต Johnsonms สำหรับ diagnosis/แนวคิด algebraic cancellation ที่มีอยู่ก่อน รายละเอียด `PRIOR_CODE_AUDIT.md`

## ชุดยืนยันและเหตุการณ์ที่ต้องเก็บไว้

Protocol V1 หยุดหลังสี่ cases เพราะตรวจ K-gradient ด้วย bitwise equality ซึ่งไม่เหมาะกับ GQA atomic reductions ของ kernel เดิม การรันวิธีเดิมซ้ำยืนยัน K variation ได้ จึงเก็บ V1 เป็น interrupted development แล้วแก้เฉพาะ harness ใช้ seeds ใหม่ 211/223/227 เป็น V2 ไม่เปลี่ยนสูตรหรือ threshold

V2: 36 cases = 3 sequence lengths × 2 initial-state choices × 2 objectives × 3 seeds; B=1, H=2, Hqk=1, N=64, P=32, 16 rotated pairs, chunk=64, bf16 values และ FP32 scheduling/state inputs ใช้ reference จาก **caller inputs จริงและ cotangents ที่ปัดเป็น dtype เดียวกับที่ GPU รับ** ทำให้ไม่ปน cotangent mismatch แบบการทดลองเก่า

ทั้งสามความยาว 257/2049/8193 มี last chunk เพียงหนึ่ง token จึงเป็นการตรวจ ragged-length กรณีเฉพาะ ไม่ใช่ทุก chunk remainder ตัวเลือกนี้ทำให้ last-chunk estimator เดิมมีพจน์ที่คลาดให้สะสมเพียงน้อย ต่างจาก development ที่ last chunk เต็ม 64 tokens

## ผลที่วัดได้

ตัวเลขคือมัธยฐาน angle-gradient normwise relative L2 ของหก cases (3 seeds × initial-state choices) ต่อแถว ไม่ใช่ผลการฝึกโมเดลหรือ perplexity

| T | Objective | FP64 sum of noisy local terms | Boundary / last-chunk κ | Boundary / direct terminal κ |
|---:|---|---:|---:|---:|
| 257 | output only | 0.02632 | 0.01496 | 0.01441 |
| 257 | output + final states | 0.00926 | 0.00564 | 0.00560 |
| 2049 | output only | 0.07365 | 0.01505 | 0.01493 |
| 2049 | output + final states | 0.02671 | 0.00591 | 0.00669 |
| 8193 | output only | 0.17007 | 0.01589 | 0.01575 |
| 8193 | output + final states | 0.05278 | 0.00643 | 0.00659 |

ผลรวม long cases (T≥2049):

- มัธยฐาน baseline/direct-terminal error ratio = **7.147×**, ผ่านเกณฑ์ ≥2 ของ boundary benefit
- มัธยฐาน legacy/direct-terminal error ratio = **1.000×**, **ไม่ผ่าน** เกณฑ์ ≥1.2 ของประโยชน์ที่เพิ่มจาก direct terminal
- Direct terminal แย่กว่า legacy 17/36 cases; แย่ขึ้น ≥20% จำนวน 1 cases ต้องเก็บเป็นข้อจำกัด ไม่ตัดทิ้ง
- Worst direct-terminal gradient error 0.01742; baseline worst 0.18616
- Max unrelated K repeat/variant variation relL2 0.000654; Q และ forward ผ่าน bitwise equality ของ harness ทุก case

![Gradient validation](iteration02_gradient_errors.png)

## คำตัดสินตามเกณฑ์ก่อนทดลอง

1. **เดินหน้า boundary formulation ในฐานะ candidate**: ลดช่องทางสะสม noisy local terms ข้าม chunk ได้ทั้งทฤษฎีและผลใหม่ แต่ยังไม่ยืนยัน novelty หรือ superiority ต่อ strong precision/direct-adjoint baselines
2. **ปฏิเสธการแทน κ เดิมด้วย direct-terminal แบบเหมารวมว่าเป็น improvement ที่พอเพียง**: ผลดีใน development ไม่ทั่วไปถึงชุด ragged ที่ freeze ไว้ สูตรตรงยังถูกต้อง แต่ความถูกต้องไม่ได้รับรอง conditioning หรือความแม่นเหนือ estimator เดิมทุกกรณี
3. รูปแบบสุดท้ายอาจใช้ κ=0 เมื่อ objective ไม่พึ่ง final states (identity ที่พิสูจน์ได้) และศึกษาวิธีเลือก terminal evaluation เมื่อมี terminal loss โดยต้องใช้ protocol ใหม่และไม่เปลี่ยนผลรอบนี้ให้กลายเป็น success
4. ยังไม่มี timing claim ของ prototype; เก็บ full adjoint tensors ทำให้ memory/เวลาไม่ใช่ implementation สุดท้าย มี `terminal_fused/` เป็น implementation candidate ที่สร้างแล้วแต่ยังต้อง validate ก่อนใช้

## ช่องโหว่ที่ยังต้องปิดก่อนส่ง

- เปรียบเทียบ direct local-rotation adjoint และ stronger precision baselines ที่เหมาะสม ไม่ใช่เฉพาะ cumsum ที่ทราบว่ามี cancellation problem
- ตรวจ novelty ต่อสมการและโค้ดเดิมครบในขอบเขต claim; state-adjoint differentiation ไม่ใช่เครื่องมือใหม่
- Stress tests: conditioning ของ terminal contractions, slow decay/long memory, chunk remainder อื่น, full rotation, GQA/shape variations; varlen/MIMO ให้จำกัด scope หากไม่ได้รองรับ
- Validate และวัด fused implementation รวม terminal reduction, memory และ setup จริง; no-overhead/no-length-growth เป็นคำอ้างที่ยังใช้ไม่ได้
- เชื่อมผลกับ numerical application และเขียน focused English manuscript พร้อม references/source/reproduction instructions เมื่อ evidence map ผ่าน

ข้อมูลดิบและโค้ดอยู่ใน folder เดียวกัน: `terminal_gpu_heldout.json`, `terminal_heldout_metrics.csv`, `terminal_heldout_summary.json`, `test_terminal_gpu_heldout.py`, `manual_adjoint_reference.py`, protocols V1/V2 และไฟล์ interrupted/atomic diagnostics

**ไม่ควรส่งร่าง broad Lag/ODE เดิมเป็น new algorithm และไม่ควรเรียกผลรอบนี้ว่าพร้อม IEEE แล้ว** Gate readiness ยังเปิด งานรอบต่อไปต้องทดสอบ candidate ที่ระบุให้แคบลง โดยรักษา negative results ทั้งสองรอบไว้
