# Direct local-coordinate adjoint baseline

Rotating-coordinate equivalence is established in Mamba-3, and reverse-mode differentiation is standard. This baseline is not presented as a new algorithm. It tests whether the boundary method has practical value beyond merely avoiding a cumulative-angle derivative representation.

The baseline uses FP32 local states and direct local-rotation derivatives, no angle cumsum and no angular reverse cumsum. It computes gradients of Q, K, V, ADT, DT, Trap, Q/K biases, Angles, D and Z. Initial state is zero and loss depends on the main output only; D/Z must be present. That scope matches the regular-training development benchmark. Inputs and output cotangents have the same quantization as competing implementations. All-state FP32 storage is charged to memory; all partial reductions and dtype conversions are charged to backward time.

First independently check every gradient against Torch FP64 autograd for small shapes and both full/partial rotation. Then use the validated checkpoint reference for larger cases. Tune V tiles4/8/16 on development before freezing final choices. Include all successful configurations in work–precision evidence; retain failures. Persistent sequential time dependence is a limitation, so do not claim this baseline represents every optimized possible local-adjoint implementation.
