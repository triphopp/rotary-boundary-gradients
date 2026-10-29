# Post-study local-adjoint FMA audit

The frozen local baseline disabled floating-point fusion to simplify arithmetic validation. A subsequent audit enabled FMA without changing the candidate or replacing the frozen-study results. It passed18 all-gradient implementation cases, then tested all4primary shapes with existing seed1013. Vtile8 was retained; a development4/8/16 sweep also favored8. This is a supplementary optimization audit on already-seen cases, not another held-out study.

| Shape | FMA angle error | FMA backward ms | Frozen local ms | Frozen TF32 ms |
|---|---:|---:|---:|---:|
| 0 | 2.586e-07 | 2.728 | 4.894 | 0.800 |
| 1 | 2.575e-07 | 6.204 | 7.598 | 1.514 |
| 2 | 2.597e-07 | 12.733 | 12.245 | 1.413 |
| 3 | 2.625e-07 | 22.572 | 24.282 | 2.949 |

FMA improved the best development time from about7.23ms to6.12ms; the separate shape audit includes one timing regression and is not a paired estimate. It remains slower than the qualifying TF32 comparator in all4checked shapes. Main-paper speedups remain the original frozen-study estimates. This audit does not prove that every possible local-coordinate kernel is slower.
