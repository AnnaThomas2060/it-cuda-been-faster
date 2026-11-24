"""Stopping early on success must not earn a lower return than continuing."""

from types import SimpleNamespace

import pytest

from src.pipeline.state import Trajectory


def trajectory_with_rewards(rewards):
    trajectory = Trajectory(trajectory_id="t", task_id="swish", prompt_mode="reference_only")
    trajectory.steps = [SimpleNamespace(reward=r, discounted_return=0.0) for r in rewards]
    return trajectory


def test_early_success_is_padded_with_terminal_reward():
    gamma = 0.9
    early = trajectory_with_rewards([1.0])  # solved at turn 0, stopped
    early.compute_returns(gamma, horizon=3)
    late = trajectory_with_rewards([0.0, 1.0, 1.0])  # solved at turn 1
    late.compute_returns(gamma)
    # Early: 1 + 0.9 + 0.81 = 2.71 > late: 0 + 0.9 + 0.81 = 1.71
    assert early.steps[0].discounted_return == pytest.approx(1.0 + 0.9 + 0.81)
    assert early.steps[0].discounted_return > late.steps[0].discounted_return


def test_no_horizon_keeps_plain_discounting():
    trajectory = trajectory_with_rewards([0.0, 2.0])
    trajectory.compute_returns(0.5)
    assert [s.discounted_return for s in trajectory.steps] == [1.0, 2.0]
