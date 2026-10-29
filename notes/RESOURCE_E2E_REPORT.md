# Warmed memory and complete-primitive measurements

Four existing primary shapes, seed1013, seven methods; each measurement process is independent. Full forward/backward includes dtype promotion and caller-gradient conversion. No model projection, optimizer or training iteration is implied.

| Shape | Method | Extra peak MiB | Captured full primitive ms | Ordinary wall ms |
|---|---|---:|---:|---:|
| 0 | BF16 suffix | 12.123 | 0.5154 | 2.0609 |
| 0 | FP32 intermediates | 14.371 | 0.5325 | 2.8375 |
| 0 | Legacy boundary | 12.123 | 0.5219 | 1.4076 |
| 0 | Boundary policy | 12.138 | 0.5252 | 2.3265 |
| 0 | FP32 / TF32 | 31.908 | 1.1244 | 1.8326 |
| 0 | FP32 / TF32x3 | 29.910 | 7.9807 | 8.4360 |
| 0 | Local FP32 adjoint | 176.244 | 5.1786 | 5.4523 |
| 1 | BF16 suffix | 49.513 | 1.1475 | 2.0504 |
| 1 | FP32 intermediates | 59.513 | 1.6964 | 3.8882 |
| 1 | Legacy boundary | 49.513 | 1.1741 | 1.6654 |
| 1 | Boundary policy | 49.575 | 1.0977 | 1.5154 |
| 1 | FP32 / TF32 | 131.648 | 2.2241 | 2.5914 |
| 1 | FP32 / TF32x3 | 123.646 | 18.8719 | 19.4806 |
| 1 | Local FP32 adjoint | 707.009 | 10.1396 | 10.4498 |
| 2 | BF16 suffix | 75.097 | 1.7304 | 2.2866 |
| 2 | FP32 intermediates | 95.102 | 1.9069 | 2.2783 |
| 2 | Legacy boundary | 75.097 | 1.7243 | 2.1087 |
| 2 | Boundary policy | 75.223 | 1.6432 | 2.0388 |
| 2 | FP32 / TF32 | 179.440 | 2.7231 | 3.1783 |
| 2 | FP32 / TF32x3 | 171.436 | 20.6114 | 21.5841 |
| 2 | Local FP32 adjoint | 746.271 | 17.6913 | 17.9196 |
| 3 | BF16 suffix | 150.019 | 4.5363 | 5.0069 |
| 3 | FP32 intermediates | 190.019 | 3.5583 | 3.7845 |
| 3 | Legacy boundary | 150.019 | 4.3684 | 5.4080 |
| 3 | Boundary policy | 150.269 | 3.4767 | 3.7245 |
| 3 | FP32 / TF32 | 358.535 | 4.7517 | 5.4206 |
| 3 | FP32 / TF32x3 | 342.527 | 41.6246 | 41.7625 |
| 3 | Local FP32 adjoint | 1474.009 | 39.9288 | 40.0503 |
