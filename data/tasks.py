from math import prod
from typing import Any, Dict, List, Optional, Tuple


def swish_ref(x):
    import torch

    return x * torch.sigmoid(x)


def column_max_ref(x):
    import torch

    return torch.max(x, dim=0).values


def channel_rmsnorm_ref(x, eps: float = 1e-5):
    import torch

    variance = torch.mean(x * x, dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps)


def avg_pool2d_ref(x):
    import torch

    return torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)


def softmax_ref(x):
    import torch

    return torch.softmax(x, dim=-1)


def cumsum_ref(x):
    import torch

    return torch.cumsum(x, dim=-1)


def default_input_gen(
    shape: Tuple[int, ...],
    device: str = "cuda",
    seed: int = 42,
    generator: Optional[Any] = None,
):
    import torch

    if generator is None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    return torch.randn(shape, dtype=torch.float32, device=device, generator=generator)


def elementwise_io_bytes(shape: Tuple[int, ...]) -> int:
    return 2 * prod(shape) * 4


def reduction_io_bytes(shape: Tuple[int, ...]) -> int:
    return prod(shape) * 4 + shape[-1] * 4


def pooling_io_bytes(shape: Tuple[int, ...]) -> int:
    n, c, h, w = shape
    return prod(shape) * 4 + n * c * (h // 2) * (w // 2) * 4


TASKS: Dict[str, Dict[str, Any]] = {
    "swish": {
        "name": "Swish Activation",
        "description": "Elementwise operation f(x) = x * sigmoid(x).",
        "shapes": [(256, 1024), (512, 2048), (1024, 4096)],
        "fn": swish_ref,
        "input_gen": default_input_gen,
        "exact_match": False,
        "is_training": True,
        "rtol": 1e-4,
        "atol": 1e-4,
        "io_bytes": elementwise_io_bytes,
        "c_signature": "torch::Tensor run_cuda(torch::Tensor x)",
    },
    "column_max": {
        "name": "Column Maximum",
        "description": "Reduction computing maximum values along dimension 0.",
        "shapes": [(256, 1024), (512, 1024), (1024, 2048)],
        "fn": column_max_ref,
        "input_gen": default_input_gen,
        "exact_match": True,
        "is_training": True,
        "rtol": 0.0,
        "atol": 0.0,
        "io_bytes": reduction_io_bytes,
        "c_signature": "torch::Tensor run_cuda(torch::Tensor x)",
    },
    "channel_rmsnorm": {
        "name": "Channel RMSNorm",
        "description": "Root mean square normalization along the trailing dimension.",
        "shapes": [(32, 512), (64, 1024), (128, 2048)],
        "fn": channel_rmsnorm_ref,
        "input_gen": default_input_gen,
        "exact_match": False,
        "is_training": True,
        "rtol": 1e-4,
        "atol": 1e-4,
        "io_bytes": elementwise_io_bytes,
        "c_signature": "torch::Tensor run_cuda(torch::Tensor x)",
    },
    "average_pooling_2d": {
        "name": "Average Pooling 2D",
        "description": "Two dimensional average pooling with kernel size 2 and stride 2.",
        "shapes": [(8, 64, 32, 32), (16, 128, 64, 64), (32, 256, 64, 64)],
        "fn": avg_pool2d_ref,
        "input_gen": default_input_gen,
        "exact_match": False,
        "is_training": False,
        "rtol": 1e-4,
        "atol": 1e-4,
        "io_bytes": pooling_io_bytes,
        "c_signature": "torch::Tensor run_cuda(torch::Tensor x)",
    },
    "softmax": {
        "name": "Softmax",
        "description": "Softmax normalization along the trailing dimension.",
        "shapes": [(64, 1024), (128, 2048), (256, 4096)],
        "fn": softmax_ref,
        "input_gen": default_input_gen,
        "exact_match": False,
        "is_training": False,
        "rtol": 1e-4,
        "atol": 1e-4,
        "io_bytes": elementwise_io_bytes,
        "c_signature": "torch::Tensor run_cuda(torch::Tensor x)",
    },
    "cumsum": {
        "name": "Cumulative Sum",
        "description": "Cumulative sum scan along the trailing dimension.",
        "shapes": [(64, 1024), (128, 2048), (256, 4096)],
        "fn": cumsum_ref,
        "input_gen": default_input_gen,
        "exact_match": False,
        "is_training": False,
        "rtol": 1e-4,
        "atol": 1e-4,
        "io_bytes": elementwise_io_bytes,
        "c_signature": "torch::Tensor run_cuda(torch::Tensor x)",
    },
}


def get_task(task_id: str) -> Dict[str, Any]:
    if task_id not in TASKS:
        raise KeyError(f"Task {task_id} not registered in TASKS.")
    return TASKS[task_id]


def list_task_ids(training_only: bool = False, holdout_only: bool = False) -> List[str]:
    if training_only and holdout_only:
        raise ValueError("training_only and holdout_only are mutually exclusive")
    if training_only:
        return [tid for tid, data in TASKS.items() if data.get("is_training", False)]
    if holdout_only:
        return [tid for tid, data in TASKS.items() if not data.get("is_training", False)]
    return list(TASKS.keys())


def get_task_tolerance(task_id: str) -> Tuple[float, float, bool]:
    task = get_task(task_id)
    return (
        float(task.get("rtol", 1e-4)),
        float(task.get("atol", 1e-4)),
        bool(task.get("exact_match", False)),
    )


def task_io_bytes(task_id: str, shape: Tuple[int, ...]) -> int:
    task = get_task(task_id)
    return int(task["io_bytes"](shape))
