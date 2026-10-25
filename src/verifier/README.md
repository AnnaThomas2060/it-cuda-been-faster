# Verifier Vertical

Owner: Pramod Goyal

## Overview
This module provides sandboxed compilation and hardware execution profiling for generated CUDA kernels.

## Contract Interface
Implementations in this package consume `KernelEvalRequest` and return `KernelEvalResponse` objects defined in `src.contracts`.

### Responsibilities
- Compile generated CUDA code via `nvcc` or `torch.utils.cpp_extension.load_inline`.
- Validate kernel output correctness against baseline PyTorch references.
- Measure execution latency using CUDA Events.
- Record achieved memory bandwidth in GB/s.
