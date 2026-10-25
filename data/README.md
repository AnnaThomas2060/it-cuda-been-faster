# Datasets

This directory manages dataset definitions and caches for kernel synthesis tasks.

## Structure
- `data/raw/`: Ignored raw benchmark repositories and source files.
- `data/cache/`: Processed tokenized trajectories and prompt templates.
- `data/cuda_rl_corpus_500/`: Training corpus derived from KernelBench tasks.

## Benchmark Sources
- KernelBench: 250 distinct PyTorch-to-CUDA operator tasks across 3 difficulty tiers.
- Terminal-Bench: Realistic workload evaluation suite for out-of-distribution kernels.
