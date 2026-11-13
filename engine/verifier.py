from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from data.tasks import get_task


@dataclass
class VerificationResult:
    correct: bool
    status: str
    message: str
    max_diff: float = 0.0
    input_mutated: bool = False
    output_aliases_input: bool = False
    checked_shapes: List[Tuple[int, ...]] = field(default_factory=list)
    checked_seeds: List[int] = field(default_factory=list)


def verify_kernel(
    module: Any,
    task_id: str,
    device: str = "cuda",
    seeds: Optional[List[int]] = None,
    shapes: Optional[List[Tuple[int, ...]]] = None,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
) -> VerificationResult:
    try:
        import torch
    except ImportError:
        return VerificationResult(
            False, "runtime_unavailable", "PyTorch is required for numerical validation."
        )

    if not torch.cuda.is_available():
        return VerificationResult(
            False, "runtime_unavailable", "CUDA device is not available for execution."
        )

    task = get_task(task_id)
    shapes = shapes or list(task["shapes"])
    seeds = seeds or [101, 202, 303, 404]
    ref_fn = task["fn"]
    input_gen = task["input_gen"]
    exact_match = bool(task.get("exact_match", False))
    rtol = float(task.get("rtol", 1e-4) if rtol is None else rtol)
    atol = float(task.get("atol", 1e-4) if atol is None else atol)

    try:
        with torch.cuda.device(device):
            for shape in shapes:
                for seed in seeds:
                    inp = input_gen(shape, device=device, seed=seed)
                    inp_for_ref = inp.clone()
                    inp_for_cand = inp.clone()

                    ref_out = ref_fn(inp_for_ref)
                    cand_out = module.run_cuda(inp_for_cand)
                    torch.cuda.synchronize()

                    if not torch.equal(inp_for_cand, inp):
                        return VerificationResult(
                            False,
                            "incorrect",
                            f"In-place tensor mutation detected for shape {shape}.",
                            input_mutated=True,
                            checked_shapes=[shape],
                            checked_seeds=[seed],
                        )

                    if not isinstance(cand_out, torch.Tensor):
                        return VerificationResult(
                            False,
                            "incorrect",
                            f"Expected torch.Tensor output, got {type(cand_out).__name__}.",
                            checked_shapes=[shape],
                            checked_seeds=[seed],
                        )

                    if not cand_out.is_cuda:
                        return VerificationResult(
                            False,
                            "incorrect",
                            "Candidate output tensor must be on CUDA device, got "
                            f"{cand_out.device}.",
                            checked_shapes=[shape],
                            checked_seeds=[seed],
                        )

                    if cand_out.shape != ref_out.shape:
                        return VerificationResult(
                            False,
                            "incorrect",
                            f"Shape mismatch: expected {ref_out.shape}, got {cand_out.shape}.",
                            checked_shapes=[shape],
                            checked_seeds=[seed],
                        )

                    if cand_out.dtype != ref_out.dtype:
                        return VerificationResult(
                            False,
                            "incorrect",
                            f"Dtype mismatch: expected {ref_out.dtype}, got {cand_out.dtype}.",
                            checked_shapes=[shape],
                            checked_seeds=[seed],
                        )

                    if cand_out.data_ptr() == inp_for_cand.data_ptr():
                        return VerificationResult(
                            False,
                            "incorrect",
                            "Candidate output aliases the input tensor.",
                            output_aliases_input=True,
                            checked_shapes=[shape],
                            checked_seeds=[seed],
                        )

                    if not torch.isfinite(cand_out).all():
                        return VerificationResult(
                            False,
                            "incorrect",
                            f"Non-finite values detected for shape {shape}.",
                            checked_shapes=[shape],
                            checked_seeds=[seed],
                        )

                    if exact_match:
                        if not torch.equal(cand_out, ref_out):
                            diff = torch.max(torch.abs(cand_out - ref_out)).item()
                            return VerificationResult(
                                False,
                                "incorrect",
                                f"Exact match failed for shape {shape}, max diff: {diff:.6f}.",
                                max_diff=float(diff),
                                checked_shapes=[shape],
                                checked_seeds=[seed],
                            )
                    elif not torch.allclose(cand_out, ref_out, rtol=rtol, atol=atol):
                        diff = torch.max(torch.abs(cand_out - ref_out)).item()
                        return VerificationResult(
                            False,
                            "incorrect",
                            f"Numerical tolerance exceeded for shape {shape}, diff: {diff:.6f}.",
                            max_diff=float(diff),
                            checked_shapes=[shape],
                            checked_seeds=[seed],
                        )
    except Exception as exc:
        return VerificationResult(
            False,
            "runtime_error",
            f"Runtime error during validation: {str(exc)[:400]}",
        )

    return VerificationResult(
        True,
        "ok",
        "Numerical equivalence verified.",
        checked_shapes=[tuple(shape) for shape in shapes],
        checked_seeds=list(seeds),
    )


def verify_numerical_correctness(module: Any, task_id: str) -> Tuple[bool, str]:
    result = verify_kernel(module, task_id)
    return result.correct, result.message
