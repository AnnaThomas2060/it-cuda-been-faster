import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from engine.utils import now_iso, sha256_text


@dataclass
class ArchiveEntry:
    task_id: str
    code: str
    speedup: float
    correct: bool
    status: str
    prompt_mode: str = "reference_only"
    turn_index: int = 0
    timestamp: str = field(default_factory=now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def code_hash(self) -> str:
        return sha256_text(self.code)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["code_hash"] = self.code_hash
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ArchiveEntry":
        allowed = {key: value for key, value in payload.items() if key in cls.__dataclass_fields__}
        return cls(**allowed)


class ContrastiveArchive:
    def __init__(
        self,
        path: str,
        capacity_per_task: int = 16,
        min_speedup_margin_delta: float = 0.25,
    ) -> None:
        self.path = Path(path)
        self.capacity_per_task = max(4, capacity_per_task)
        self.min_speedup_margin_delta = min_speedup_margin_delta
        self.entries: Dict[str, List[ArchiveEntry]] = {}
        if self.path.exists():
            self.load()

    def add(
        self,
        task_id: str,
        code: str,
        speedup: float,
        correct: bool,
        status: str,
        prompt_mode: str = "reference_only",
        turn_index: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[ArchiveEntry]:
        if not code.strip():
            return None
        entry = ArchiveEntry(
            task_id=task_id,
            code=code,
            speedup=float(speedup),
            correct=bool(correct),
            status=status,
            prompt_mode=prompt_mode,
            turn_index=turn_index,
            metadata=metadata or {},
        )
        bucket = self.entries.setdefault(task_id, [])
        if any(existing.code_hash == entry.code_hash for existing in bucket):
            return None
        bucket.append(entry)
        if len(bucket) > self.capacity_per_task:
            ranked = sorted(bucket, key=lambda item: (item.correct, item.speedup), reverse=True)
            keep = max(1, self.capacity_per_task // 2)
            winners = ranked[:keep]
            losers = ranked[-keep:]
            merged = {item.code_hash: item for item in winners + losers}
            self.entries[task_id] = list(merged.values())
        return entry

    def winners(self, task_id: str, limit: int = 4) -> List[ArchiveEntry]:
        bucket = [entry for entry in self.entries.get(task_id, []) if entry.correct]
        bucket.sort(key=lambda entry: entry.speedup, reverse=True)
        return bucket[:limit]

    def losers(self, task_id: str, limit: int = 4) -> List[ArchiveEntry]:
        bucket = list(self.entries.get(task_id, []))
        bucket.sort(key=lambda entry: (entry.correct, entry.speedup))
        return bucket[:limit]

    def sample_pair(
        self,
        task_id: str,
        rng: Optional[random.Random] = None,
    ) -> Optional[Tuple[ArchiveEntry, ArchiveEntry]]:
        rng = rng or random
        winners = self.winners(task_id)
        losers = self.losers(task_id)
        if not winners or not losers:
            return None
        for _ in range(8):
            winner = rng.choice(winners)
            loser = rng.choice(losers)
            if winner.code_hash == loser.code_hash:
                continue
            if not loser.correct:
                return winner, loser
            if winner.speedup >= loser.speedup + self.min_speedup_margin_delta:
                return winner, loser
        return None

    def all_pairs(
        self, max_pairs: int = 8, seed: int = 0
    ) -> List[Tuple[str, ArchiveEntry, ArchiveEntry]]:
        rng = random.Random(seed)
        pairs = []
        task_ids = sorted(self.entries)
        rng.shuffle(task_ids)
        for task_id in task_ids:
            pair = self.sample_pair(task_id, rng)
            if pair is not None:
                pairs.append((task_id, pair[0], pair[1]))
            if len(pairs) >= max_pairs:
                break
        return pairs

    def summary(self) -> Dict[str, Any]:
        return {
            task_id: {
                "count": len(entries),
                "correct": sum(1 for entry in entries if entry.correct),
                "best_speedup": max((entry.speedup for entry in entries), default=0.0),
            }
            for task_id, entries in self.entries.items()
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            for task_id in sorted(self.entries):
                for entry in self.entries[task_id]:
                    handle.write(json.dumps(entry.to_dict(), ensure_ascii=True) + "\n")
        tmp.replace(self.path)

    def load(self) -> None:
        self.entries = {}
        with open(self.path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                entry = ArchiveEntry.from_dict(json.loads(line))
                self.entries.setdefault(entry.task_id, []).append(entry)
