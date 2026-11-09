from typing import Any, Tuple

from data.tasks import get_task

# Numerical verification suite testing equivalence between CUDA modules and PyTorch baselines.
# Tests multiple tensor dimensions and random seeds while verifying input immutability.
# Discrepancies beyond tolerance or in place mutations trigger immediate validation failure.

def verify_numerical_correctness(module: Any, task_id: str) -> Tuple[bool, str]:
    try:
        import torch
    except ImportError:
        return False, "PyTorch is required for numerical validation."

    if not torch.cuda.is_available():
        return False, "CUDA device is not available for execution."

    task = get_task(task_id)
    shapes = task["shapes"]
    ref_fn = task["fn"]
    input_gen = task["input_gen"]
    exact_match = task.get("exact_match", False)

    seeds = [101, 202, 303, 404]
    for shape in shapes:
        for seed in seeds:
            try:
                inp = input_gen(shape, device="cuda", seed=seed)
                inp_for_ref = inp.clone()
                inp_for_cand = inp.clone()

                ref_out = ref_fn(inp_for_ref)
                cand_out = module.run_cuda(inp_for_cand)
                torch.cuda.synchronize()

                # Verify input was not modified in-place
                if not torch.equal(inp_for_cand, inp):
                    return False, f"In-place tensor mutation detected for shape {shape}."

                if not isinstance(cand_out, torch.Tensor):
                    return False, f"Expected torch.Tensor output, got {type(cand_out).__name__}."

                if not cand_out.is_cuda:
                    dev = cand_out.device
                    return False, f"Candidate output tensor must be on CUDA device, got {dev}."

                if cand_out.shape != ref_out.shape:
                    return (
                        False,
                        f"Shape mismatch: expected {ref_out.shape}, got {cand_out.shape}.",
                    )

                if cand_out.dtype != ref_out.dtype:
                    return (
                        False,
                        f"Dtype mismatch: expected {ref_out.dtype}, got {cand_out.dtype}.",
                    )

                if not torch.isfinite(cand_out).all():
                    return (
                        False,
                        f"Non-finite values (NaN or Inf) detected in candidate for shape {shape}.",
                    )

                if exact_match:
                    if not torch.equal(cand_out, ref_out):
                        diff = torch.max(torch.abs(cand_out - ref_out)).item()
                        return False, f"Exact match failed for shape {shape}, max diff: {diff:.6f}."
                else:
                    if not torch.allclose(cand_out, ref_out, rtol=1e-4, atol=1e-4):
                        diff = torch.max(torch.abs(cand_out - ref_out)).item()
                        return (
                            False,
                            f"Numerical tolerance exceeded for shape {shape}, diff: {diff:.6f}.",
                        )

            except Exception as e:
                return False, f"Runtime error during validation: {str(e)[:300]}"

    return True, "Numerical equivalence verified."
