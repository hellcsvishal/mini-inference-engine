# Efficient Neural Inference: From FP32 to SIMD-Accelerated Execution


This project investigates the impact of numerical representation and system-level optimizations on neural network inference performance.

We implement and compare multiple computation paradigms:

- FP32 (baseline)
- INT8 quantization
- Ternary computation {-1, 0, +1}
- Loop-unrolled execution
- SIMD (AVX) vectorization

The study demonstrates that reducing arithmetic complexity alone does not guarantee performance gains. Instead, memory access patterns, branching behavior, and hardware-aware optimizations dominate real-world efficiency.


FP32 → INT8 → Ternary → Memory Optimization → Loop Unrolling → SIMD

```
            Input Vector (x)
                    │
                    ▼
          ┌───────────────────┐
          │ Weight Matrix (W) │
          └───────────────────┘
                    │
        ┌───────────┼────────────┐
        ▼           ▼            ▼
     FP32        INT8        Ternary
   (baseline)  (quantized) (no multiply)
        │           │            │
        └───────────┼────────────┘
                    ▼
        Optimized Execution Layer
        (Unrolling + SIMD AVX)
                    │
                    ▼
               Output Vector
```

# Implementation Details
1. FP32 Baseline

- Standard matrix-vector multiplication using floating-point arithmetic.

2. INT8 Quantization

- Weights and inputs are quantized to 8-bit integers. Computation uses integer arithmetic.

3. Ternary Representation

- Weights are restricted to {-1, 0, +1}. Multiplication is replaced with addition/subtraction.

4. Memory Optimization

- Matrix is flattened into a contiguous array to improve cache locality.

5. Loop Unrolling

- Manual unrolling reduces loop overhead and improves instruction-level parallelism.

6. SIMD (AVX)

- Vectorized computation using 256-bit registers processes 8 floating-point values per instruction.


# Benchmark Results

| Method   | Time (sec)|
|----------|-----------|
| FP32     | 0.0396    |
| Unrolled | 0.0388    |
| AVX      | 0.0328    |



# Key Findings
1. Quantization alone does not guarantee speedup.
2. Branching introduces significant performance penalties.
3. Memory layout has a major impact on execution time.
4. SIMD provides measurable acceleration (~17%).
5. Performance bottleneck shifts from compute to memory.


# Critical Insight
- Theoretical reductions in arithmetic complexity (e.g., ternary weights eliminating multiplications) do not directly translate to practical speedups.
- Modern CPUs are highly optimized for floating-point operations, and without careful memory and hardware-aware optimizations, reduced-precision models may perform worse in practice.


# Future Work
- Bit-packing for 1-bit/ternary weights
- Popcount-based computation kernels
- AVX-512 / advanced SIMD
- Integration with real transformer layers


# Conclusion
This project demonstrates that efficient AI is fundamentally a systems problem, not just a mathematical one.

Achieving real performance gains requires co-design across:
- numerical representation
- memory layout
- execution model
- hardware capabilities


