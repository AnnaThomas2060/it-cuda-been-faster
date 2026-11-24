import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from engine.utils import JsonlWriter, now_iso, sha256_text
from src.contracts import KernelEvalRequest, KernelEvalResponse, TelemetryRecord
from src.rewards.archive import ContrastiveArchive


@dataclass
class RewardResult:
    task_id: str
    code: str
    reward: float
    status: str
    response: KernelEvalResponse
    speedup: float = 0.0
    taxonomy: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


def compute_shaped_reward(
    response: KernelEvalResponse,
    reward_mode: str = "shaped",
    correctness_bonus: float = 0.30,
    bandwidth_weight: float = 0.10,
    reward_clip: float = 2.0,
) -> float:
    if not response.compiled or not response.correct or not response.stable:
        return 0.0
    if response.status in {
        "fake_gain",
        "implausible_speedup",
        "cache_or_incorrect",
        "measurement_too_small",
        "unstable",
    }:
        return 0.0
    speedup = float(response.speedup or 0.0)
    if reward_mode == "speedup":
        return max(0.0, min(speedup, reward_clip))
    speedup_gain = math.log1p(max(speedup - 1.0, 0.0))
    bandwidth_delta = max(-1.0, min(1.0, float(response.bandwidth_delta or 0.0)))
    reward = correctness_bonus + speedup_gain + bandwidth_weight * bandwidth_delta
    return max(0.0, min(reward, reward_clip))


def assign_discounted_returns(rewards: List[float], gamma: float) -> List[float]:
    returns = [0.0] * len(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = rewards[index] + gamma * running
        returns[index] = running
    return returns


def group_normalize(values: List[float], epsilon: float = 1e-6) -> List[float]:
    if len(values) < 2:
        return [0.0 for _ in values]
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    if std < epsilon:
        return [0.0 for _ in values]
    return [(value - mean_value) / (std + epsilon) for value in values]


def _device_index(device: str) -> int:
    if ":" in device:
        try:
            return int(device.split(":", 1)[1])
        except ValueError:
            return 0
    return 0


class RewardEngine:
    def __init__(
        self,
        config: Any,
        dispatcher: Any,
        archive: Optional[ContrastiveArchive] = None,
        telemetry_path: Optional[str] = None,
    ) -> None:
        self.config = config
        self.dispatcher = dispatcher
        self.archive = archive
        self.telemetry = JsonlWriter(telemetry_path or config.telemetry_path)

    def evaluate_code(
        self,
        code: str,
        task_id: str,
        turn_index: int = 0,
        prompt_mode: str = "reference_only",
        run_sanitizer: bool = False,
        raw_completion: Optional[str] = None,
        trajectory_id: str = "",
        prompt_text: Optional[str] = None,
    ) -> RewardResult:
        request = KernelEvalRequest(
            task_id=task_id,
            code=code,
            turn_index=turn_index,
            timeout_seconds=float(
                self.config.compile_timeout_sec
                + self.config.verify_timeout_sec
                + self.config.benchmark_timeout_sec
            ),
            device_id=_device_index(self.config.verifier_device),
            run_sanitizer=run_sanitizer or self.config.enable_sanitizer,
            metadata={"prompt_mode": prompt_mode},
        )
        response = self.dispatcher.evaluate(request)
        reward = compute_shaped_reward(
            response,
            reward_mode=self.config.reward_mode,
            correctness_bonus=self.config.correctness_bonus,
            bandwidth_weight=self.config.bandwidth_weight,
            reward_clip=self.config.reward_clip,
        )
        status = response.status
        if response.compiled and not response.correct:
            status = "incorrect"
        elif not response.compiled and status in {"", "unknown"}:
            status = "compile_error"

        if self.archive is not None and code.strip():
            self.archive.add(
                task_id=task_id,
                code=code,
                speedup=float(response.speedup or 0.0),
                correct=bool(response.correct and response.stable),
                status=status,
                prompt_mode=prompt_mode,
                turn_index=turn_index,
                metadata={
                    "variance": response.variance,
                    "bandwidth_delta": response.bandwidth_delta,
                    "raw_completion": (raw_completion or "")[:12000],
                    # The exact prompt this completion was sampled from, so
                    # the DPO term scores it in the same context.
                    "prompt_text": prompt_text or "",
                },
            )

        result = RewardResult(
            task_id=task_id,
            code=code,
            reward=reward,
            status=status,
            response=response,
            speedup=float(response.speedup or 0.0),
            taxonomy=status,
        )
        self._write_telemetry(
            result,
            turn_index,
            prompt_mode,
            trajectory_id=trajectory_id or uuid.uuid4().hex,
        )
        return result

    def _write_telemetry(
        self,
        result: RewardResult,
        turn_index: int,
        prompt_mode: str,
        trajectory_id: str,
    ) -> None:
        response = result.response
        record = TelemetryRecord(
            trajectory_id=trajectory_id,
            task_id=result.task_id,
            turn_index=turn_index,
            speedup=response.speedup,
            reward=result.reward,
            compiled=response.compiled,
            correct=response.correct,
            memory_bandwidth_gbps=response.memory_bandwidth_gbps,
            timestamp=now_iso(),
            extra={
                "prompt_mode": prompt_mode,
                "code_hash": sha256_text(result.code),
                "status": result.status,
                "stable": response.stable,
                "variance": response.variance,
                "baseline_bandwidth_gbps": response.baseline_bandwidth_gbps,
                "bandwidth_delta": response.bandwidth_delta,
                "compile_time_sec": response.compile_time_sec,
                "benchmark_time_sec": response.benchmark_time_sec,
                "device_name": response.device_name,
                "failure_stage": response.failure_stage,
                "error_message": response.error_message,
            },
        )
        payload = record.model_dump() if hasattr(record, "model_dump") else record.dict()
        self.telemetry.write(payload)

    def close(self) -> None:
        close = getattr(self.dispatcher, "close", None)
        if close is not None:
            close()
