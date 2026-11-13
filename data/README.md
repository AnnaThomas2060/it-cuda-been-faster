# Task and Teacher Data

This directory defines the benchmark tasks and the curated teacher kernels used for contrastive prompts.

## Files

- `tasks.py`: the task registry. Each entry contains a display name, description, C++ signature, evaluation shapes, a PyTorch reference function, an input generator, tolerances, exact match behavior, a training or holdout flag, and an I/O byte estimate used for bandwidth checks.
- `example_store.py`: stored teacher kernels per task. Entries include a name, description, default speedup, code, and a verified flag. Holdout tasks have teachers so Arm A receives contrastive context on both splits.
- `teacher_measurements.json`: optional measured speedups written by `scripts/verify_teachers.py`. When present, these values override the placeholders in `example_store.py` and mark the entry as verified.

## Tasks

Training tasks:

- `swish`: elementwise `x * sigmoid(x)`.
- `column_max`: reduction along dimension 0, exact equality required.
- `channel_rmsnorm`: root mean square normalization along the trailing dimension.

Holdout tasks:

- `average_pooling_2d`: 2x2 average pooling with stride 2.
- `softmax`: softmax along the trailing dimension.
- `cumsum`: cumulative sum along the trailing dimension.

## Seed Handling

Input generation uses a local `torch.Generator` when one is provided, so verification and benchmarking do not disturb the global random state used by training.

## Teacher Measurements

Default speedups in `example_store.py` are placeholders for prompt construction and unit tests. They are not claims about measured hardware performance. Run:

```bash
python scripts/verify_teachers.py --config configs/multiturn_grpo.yaml --device cuda:0
```

on the target GPU to compile, verify, and benchmark each teacher, then write the measured overrides. Only kernels that pass compilation, numerical verification, and the timing gate are written to the measurement file.
