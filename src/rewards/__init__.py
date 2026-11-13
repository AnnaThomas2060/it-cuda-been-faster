"""Reward shaping, anti-hacking checks, and contrastive archive helpers."""

from src.rewards.archive import ArchiveEntry, ContrastiveArchive
from src.rewards.reward_engine import (
    RewardEngine,
    RewardResult,
    assign_discounted_returns,
    group_normalize,
)

__all__ = [
    "ArchiveEntry",
    "ContrastiveArchive",
    "RewardEngine",
    "RewardResult",
    "assign_discounted_returns",
    "group_normalize",
]
