"""Shared IPC and telemetry contracts.

Coordinates data exchange between the RL pipeline, verifier, and analytics.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class KernelEvalRequest(BaseModel):
    """Evaluation request payload sent to the verifier vertical."""

    task_id: str = Field(description="Unique identifier of the kernel task.")
    code: str = Field(description="Generated CUDA kernel source code.")
    baseline_code: Optional[str] = Field(
        default=None, description="Reference baseline implementation."
    )
    turn_index: int = Field(default=0, description="Trajectory turn index.")
    timeout_seconds: float = Field(
        default=30.0, description="Timeout budget for compilation and run."
    )
    device_id: int = Field(default=0, description="Target GPU device index.")
    run_sanitizer: bool = Field(default=False, description="Run compute-sanitizer on this request.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Request metadata.")


class KernelEvalResponse(BaseModel):
    """Evaluation response payload returned by the verifier vertical."""

    task_id: str = Field(description="Unique identifier of the kernel task.")
    compiled: bool = Field(description="Whether the kernel compiled successfully.")
    correct: bool = Field(description="Whether outputs match reference baseline.")
    execution_time_ms: Optional[float] = Field(
        default=None, description="Kernel runtime in milliseconds."
    )
    baseline_time_ms: Optional[float] = Field(
        default=None, description="Baseline runtime in milliseconds."
    )
    speedup: Optional[float] = Field(
        default=None, description="Calculated speedup ratio over baseline."
    )
    memory_bandwidth_gbps: Optional[float] = Field(
        default=None, description="Achieved memory bandwidth in GB/s."
    )
    baseline_bandwidth_gbps: Optional[float] = Field(
        default=None, description="Baseline memory bandwidth in GB/s."
    )
    bandwidth_delta: float = Field(default=0.0, description="Relative bandwidth change.")
    compiler_output: str = Field(default="", description="Diagnostic compiler logs and warnings.")
    error_message: Optional[str] = Field(
        default=None, description="Runtime or compilation failure message."
    )
    status: str = Field(default="unknown", description="Machine readable evaluation status.")
    stable: bool = Field(default=False, description="Whether timing passed stability checks.")
    variance: float = Field(default=0.0, description="Timing variance across blocks.")
    compile_time_sec: float = Field(default=0.0, description="Compilation wall time.")
    benchmark_time_sec: float = Field(default=0.0, description="Benchmark wall time.")
    candidate_code_hash: str = Field(default="", description="Hash of evaluated source code.")
    device_name: str = Field(default="", description="Verifier GPU name.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Supplementary details.")


class TurnStepSample(BaseModel):
    """Step sample collected within a multi-turn trajectory."""

    task_id: str = Field(default="", description="Task identifier.")
    turn_index: int = Field(description="Zero-indexed turn position.")
    prompt: str = Field(description="Prompt provided to the model.")
    thought: str = Field(default="", description="Extracted reasoning trace.")
    code: str = Field(default="", description="Generated CUDA code block.")
    response: Optional[KernelEvalResponse] = Field(
        default=None, description="Environment feedback received."
    )
    reward: float = Field(default=0.0, description="Assigned scalar reward.")
    discounted_return: float = Field(default=0.0, description="Discounted MDP return.")
    advantage: float = Field(default=0.0, description="Group-normalized advantage.")
    prompt_ids: List[int] = Field(default_factory=list, description="Tokenized prompt ids.")
    completion_ids: List[int] = Field(default_factory=list, description="Tokenized completion ids.")
    old_logprobs: List[float] = Field(
        default_factory=list, description="Behavior policy token log probabilities."
    )
    prompt_mode: str = Field(default="reference_only", description="Prompt arm.")


class TrajectoryRecord(BaseModel):
    """Serializable multi-turn trajectory."""

    trajectory_id: str
    task_id: str
    prompt_mode: str
    steps: List[TurnStepSample] = Field(default_factory=list)
    best_code: str = ""
    best_speedup: float = 0.0
    best_status: str = "unknown"
    final_reward: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TelemetryRecord(BaseModel):
    """Telemetry schema emitted for trajectory and performance analytics."""

    trajectory_id: str = Field(description="Unique identifier for rollout.")
    task_id: str = Field(description="Evaluated task identifier.")
    turn_index: int = Field(description="Turn index within trajectory.")
    speedup: Optional[float] = Field(default=None, description="Measured speedup.")
    reward: float = Field(default=0.0, description="Calculated reward.")
    compiled: bool = Field(default=False, description="Compilation status.")
    correct: bool = Field(default=False, description="Correctness status.")
    memory_bandwidth_gbps: Optional[float] = Field(default=None, description="Bandwidth in GB/s.")
    timestamp: str = Field(description="ISO-8601 creation timestamp.")
    extra: Dict[str, Any] = Field(default_factory=dict, description="Supplementary metrics.")


class MockVerifier:
    """Offline stub simulating verifier responses for local decoupling."""

    def __init__(self, simulated_speedup: float = 1.05) -> None:
        self.simulated_speedup = simulated_speedup

    def evaluate(self, request: KernelEvalRequest) -> KernelEvalResponse:
        """Return a simulated verification result for the given request."""
        has_code = bool(request.code and len(request.code.strip()) > 0)
        return KernelEvalResponse(
            task_id=request.task_id,
            compiled=has_code,
            correct=has_code,
            execution_time_ms=1.0 if has_code else None,
            baseline_time_ms=1.0 * self.simulated_speedup if has_code else None,
            speedup=self.simulated_speedup if has_code else None,
            memory_bandwidth_gbps=500.0 if has_code else None,
            baseline_bandwidth_gbps=450.0 if has_code else None,
            bandwidth_delta=0.11 if has_code else 0.0,
            compiler_output="Mock compilation successful." if has_code else "Empty code.",
            error_message=None if has_code else "Compilation failed: empty source.",
            status="ok" if has_code else "compile_error",
            stable=has_code,
        )


__all__ = [
    "KernelEvalRequest",
    "KernelEvalResponse",
    "TurnStepSample",
    "TrajectoryRecord",
    "TelemetryRecord",
    "MockVerifier",
]
