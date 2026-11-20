"""Numerical equivalence suite for a compiled candidate.

Beyond comparing outputs on a few shapes, the suite targets the failure
modes RL policies actually find (see CUDA-L1 Sec. 4 and Kevin-32B on reward
hacking):

* odd tails and tiny shapes, which expose missing bounds checks;
* reused input buffers with fresh values, which expose pointer-keyed caching;
* NaN-filled fresh allocations, which expose unwritten output elements;
* six adversarial value patterns (overflow, cancellation, ties, ...);
* a delayed write on a non-default stream, which exposes kernels that ignore
  the current stream and race ahead of their input;
* input mutation checks after every call.

Torch is imported lazily; the shape/schedule helpers run on CPU-only machines.
"""

import itertools
import math
from typing import Any, Callable, Dict, Iterable, Iterator, List, Tuple

Shape = Tuple[int, ...]

# Small shapes with odd extents, added to each task's benchmark shapes.
EDGE_SHAPES: Dict[int, List[Shape]] = {
    2: [(1, 1), (3, 33), (7, 257)],
    4: [(1, 1, 2, 2), (2, 3, 7, 9), (1, 5, 33, 17)],
}
ADVERSARIAL_SHAPE: Dict[int, Shape] = {2: (3, 33), 4: (1, 2, 6, 10)}
ADVERSARIAL_PATTERNS: Dict[str, List[float]] = {
    "zeros": [0.0],
    "all_negative": [-3.0, -5.0, -2.0],
    "repeated_values": [2.0],
    "alternating_sign": [1.0, -1.0],
    "large_and_small": [1000.0, -1000.0, 1e-8, -1e-8, 80.0, -80.0],
    "cancellation": [10000.0, 1.0, -10000.0, -1.0],
}
STREAM_DELAY_CYCLES = 100_000_000


class ValidationError(ValueError):
    """The candidate produced a wrong, mutated, or malformed result."""


def tolerances(task: Dict[str, Any]) -> Dict[str, float]:
    if task.get("exact_match"):
        return {"atol": 0.0, "rtol": 0.0}
    return {"atol": task.get("atol", 1e-4), "rtol": task.get("rtol", 1e-4)}


def validation_shapes(task: Dict[str, Any]) -> List[Shape]:
    rank = len(task["shapes"][0])
    return EDGE_SHAPES.get(rank, []) + [tuple(s) for s in task["shapes"]]


def random_case_schedule(task: Dict[str, Any], total: int, seed: int) -> Iterator[Dict[str, Any]]:
    """Deterministic cases spread evenly over shapes, grouped by shape.

    Grouping matters: consecutive same-shape cases reuse one input buffer,
    which is what catches output caching keyed on the input pointer.
    """
    shapes = validation_shapes(task)
    per_shape, remainder = divmod(total, len(shapes))
    index = 0
    for position, shape in enumerate(shapes):
        for _ in range(per_shape + (position < remainder)):
            yield {"index": index, "shape": shape, "seed": seed + index}
            index += 1


def adversarial_inputs(torch, task: Dict[str, Any], device: str) -> List[Tuple[str, Any]]:
    shape = ADVERSARIAL_SHAPE[len(task["shapes"][0])]
    n = math.prod(shape)
    cases = []
    for name, values in ADVERSARIAL_PATTERNS.items():
        tiled = (values * (n // len(values) + 1))[:n]
        tensor = torch.tensor(tiled, dtype=torch.float32, device=device).reshape(shape)
        cases.append((name, tensor))
    return cases


def validate_candidate(
    reference: Callable,
    candidate: Callable,
    cases: Iterable[Tuple[Any, ...]],
    *,
    atol: float,
    rtol: float,
) -> Dict[str, Any]:
    """Run every case; raise ValidationError on the first wrong result."""
    import torch

    buffers, signature = None, None
    passed, max_abs_error = 0, 0.0
    with torch.inference_mode():
        for case in cases:
            current = tuple((tuple(x.shape), x.dtype, x.device) for x in case)
            if current != signature:
                buffers, signature = [x.clone() for x in case], current
            else:
                for buffer, values in zip(buffers, case):
                    buffer.copy_(values)
            before = [x.clone() for x in buffers]
            expected = reference(*(x.clone() for x in before))
            actual = candidate(*buffers)

            if type(actual) is not torch.Tensor:
                raise ValidationError(f"Output is {type(actual).__name__}, not a Tensor")
            if actual.shape != expected.shape:
                raise ValidationError(
                    f"Shape mismatch: expected {tuple(expected.shape)}, got {tuple(actual.shape)}"
                )
            if actual.dtype != expected.dtype:
                raise ValidationError(
                    f"Dtype mismatch: expected {expected.dtype}, got {actual.dtype}"
                )
            if actual.device != expected.device:
                raise ValidationError(
                    f"Device mismatch: expected {expected.device}, got {actual.device}"
                )
            if actual.is_cuda:
                torch.cuda.synchronize(actual.device)
            if any(not torch.equal(b, o) for b, o in zip(buffers, before)):
                raise ValidationError("Candidate mutated its input")
            try:
                torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, equal_nan=False)
            except AssertionError as exc:
                first_line = str(exc).strip().splitlines()[0]
                raise ValidationError(
                    f"Values differ for shape {tuple(actual.shape)}: {first_line}"
                ) from exc
            if actual.numel():
                max_abs_error = max(max_abs_error, (actual - expected).abs().max().item())
            passed += 1
    if passed == 0:
        raise ValueError("validate_candidate needs at least one case")
    return {"cases_passed": passed, "max_abs_error": max_abs_error}


def nondefault_stream_case(torch, reference, candidate, desired, stream, *, atol, rtol):
    """Delay an input write on `stream`, then call the candidate on that stream.

    A kernel launched on the legacy default stream instead of the current one
    reads the stale sentinel (17.0) and fails the comparison.
    """
    buffer = torch.full_like(desired, 17.0)
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(STREAM_DELAY_CYCLES)
        buffer.copy_(desired)
        actual = candidate(buffer)
    stream.synchronize()
    torch.cuda.synchronize()
    if not torch.equal(buffer, desired):
        raise ValidationError("Candidate mutated its input on a non-default stream")
    return validate_candidate(reference, lambda _: actual, [(desired,)], atol=atol, rtol=rtol)


def run_correctness_suite(
    torch, task: Dict[str, Any], candidate: Callable, config, device: str = "cuda:0"
) -> Dict[str, Any]:
    """Random, adversarial and stream stages. Returns a JSON-safe summary."""
    reference = task["fn"]
    tol = tolerances(task)
    counts = {"random_cases": 0, "adversarial_cases": 0, "stream_cases": 0}
    stage = "random"

    previous_fill = torch.utils.deterministic.fill_uninitialized_memory
    previous_det = torch.are_deterministic_algorithms_enabled()
    # With deterministic mode on, torch.empty() returns NaN-filled memory, so a
    # kernel that skips output elements fails instead of passing by luck.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    try:
        schedule = random_case_schedule(task, config.random_cases, config.seed)
        for _, group in itertools.groupby(schedule, key=lambda item: item["shape"]):
            cases = (
                (task["input_gen"](item["shape"], device=device, seed=item["seed"]),)
                for item in group
            )
            counts["random_cases"] += validate_candidate(reference, candidate, cases, **tol)[
                "cases_passed"
            ]

        if config.adversarial_cases:
            stage = "adversarial"
            inputs = adversarial_inputs(torch, task, device)
            counts["adversarial_cases"] = validate_candidate(
                reference, candidate, ((x,) for _, x in inputs), **tol
            )["cases_passed"]

        if config.nondefault_stream_check:
            stage = "nondefault_stream"
            stream = torch.cuda.Stream(device=device)
            for _, desired in adversarial_inputs(torch, task, device):
                nondefault_stream_case(torch, reference, candidate, desired, stream, **tol)
                counts["stream_cases"] += 1

        return {"correct": True, **counts, **tol}
    except ValidationError as exc:
        return {"correct": False, "failed_stage": stage, "message": str(exc), **counts, **tol}
    finally:
        torch.use_deterministic_algorithms(previous_det)
        torch.utils.deterministic.fill_uninitialized_memory = previous_fill


def negative_controls(torch, device: str = "cuda:0") -> Dict[str, bool]:
    """Known-bad candidates that the suite must reject; guards the evaluator itself."""
    cases = [(torch.randn(3, 33, device=device),) for _ in range(2)]
    cache: Dict[int, Any] = {}

    def cached(x):
        return cache.setdefault(x.data_ptr(), torch.relu(x))

    bad = {
        "wrong_values": lambda x: torch.zeros_like(x),
        "wrong_shape": lambda x: torch.zeros(1, device=x.device),
        "wrong_dtype": lambda x: torch.relu(x).double(),
        "mutated_input": lambda x: x.relu_(),
        "cached_output": cached,
    }
    rejected = {}
    for name, fn in bad.items():
        try:
            validate_candidate(torch.relu, fn, cases, atol=1e-4, rtol=1e-4)
            rejected[name] = False
        except ValidationError:
            rejected[name] = True
    return rejected
