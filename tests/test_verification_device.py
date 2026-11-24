"""The >3x confirmation GPU is explicit and never the training or verifier GPU."""

from types import SimpleNamespace

import pytest

from src.pipeline.env_dispatcher import VerifierHarnessDispatcher

resolve = VerifierHarnessDispatcher._resolve_verification_device


def cfg(**overrides):
    base = dict(cross_gpu_validation=True, device="cuda:0", verifier_device="cuda:1",
                verification_device=None)
    return SimpleNamespace(**{**base, **overrides})


def test_unset_means_no_confirmation_not_the_training_gpu():
    assert resolve(cfg()) is None


def test_explicit_third_gpu_is_used():
    assert resolve(cfg(verification_device="cuda:2")) == 2


def test_training_or_verifier_gpu_is_rejected():
    with pytest.raises(ValueError):
        resolve(cfg(verification_device="cuda:0"))
    with pytest.raises(ValueError):
        resolve(cfg(verification_device="cuda:1"))


def test_disabled_validation_ignores_device():
    assert resolve(cfg(cross_gpu_validation=False, verification_device="cuda:2")) is None
