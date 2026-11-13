from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.contracts import KernelEvalResponse, TrajectoryRecord, TurnStepSample


def build_feedback(response: KernelEvalResponse) -> str:
    if not response.compiled:
        return f"Compilation failed. {response.error_message or response.compiler_output}"
    if not response.correct:
        return f"Numerical validation failed. {response.error_message or ''}".strip()
    if not response.stable:
        return f"Timing validation failed. {response.error_message or response.status}".strip()
    parts = [f"Correct kernel. Measured speedup {float(response.speedup or 0.0):.2f}x."]
    if response.memory_bandwidth_gbps:
        parts.append(f"Memory bandwidth {response.memory_bandwidth_gbps:.1f} GB/s.")
    if response.bandwidth_delta:
        parts.append(f"Bandwidth delta {response.bandwidth_delta:+.2%}.")
    return " ".join(parts)


@dataclass
class TurnRecord:
    turn_index: int
    prompt: str
    prompt_ids: List[int]
    completion_ids: List[int]
    completion_text: str
    old_logprobs: List[float]
    response: KernelEvalResponse
    reward: float
    code: str
    status: str
    speedup: float
    feedback: str
    thought: str = ""
    discounted_return: float = 0.0
    advantage: float = 0.0

    def to_sample(self) -> TurnStepSample:
        return TurnStepSample(
            turn_index=self.turn_index,
            prompt=self.prompt,
            thought=self.thought,
            code=self.code,
            response=self.response,
            reward=self.reward,
            discounted_return=self.discounted_return,
            advantage=self.advantage,
            prompt_ids=self.prompt_ids,
            completion_ids=self.completion_ids,
            old_logprobs=self.old_logprobs,
        )


@dataclass
class Trajectory:
    trajectory_id: str
    task_id: str
    prompt_mode: str
    steps: List[TurnRecord] = field(default_factory=list)
    best_code: str = ""
    best_speedup: float = 0.0
    best_status: str = "unknown"
    best_response: Optional[KernelEvalResponse] = None
    last_code: str = ""
    last_status: str = "unknown"
    last_speedup: float = 0.0
    final_reward: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: TurnRecord) -> None:
        self.steps.append(step)
        self.last_code = step.code
        self.last_status = step.status
        self.last_speedup = step.speedup
        if step.response.correct and step.response.stable and step.speedup > self.best_speedup:
            self.best_code = step.code
            self.best_speedup = step.speedup
            self.best_status = step.status
            self.best_response = step.response

    def history(self) -> List[Dict[str, Any]]:
        return [
            {
                "turn_index": step.turn_index,
                "status": step.status,
                "feedback": step.feedback,
            }
            for step in self.steps
        ]

    def compute_returns(self, gamma: float) -> None:
        running = 0.0
        for step in reversed(self.steps):
            running = step.reward + gamma * running
            step.discounted_return = running
        if self.steps:
            self.final_reward = self.steps[0].discounted_return

    def to_record(self) -> TrajectoryRecord:
        return TrajectoryRecord(
            trajectory_id=self.trajectory_id,
            task_id=self.task_id,
            prompt_mode=self.prompt_mode,
            steps=[step.to_sample() for step in self.steps],
            best_code=self.best_code,
            best_speedup=self.best_speedup,
            best_status=self.best_status,
            final_reward=self.final_reward,
            metadata=self.metadata,
        )
