"""Smoke tests for package imports, configuration files, and documentation."""

import importlib
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    "module_name",
    [
        "src",
        "src.contracts",
        "src.pipeline",
        "src.rewards",
        "src.trainers",
        "src.verifier",
        "src.baselines",
        "src.analytics",
    ],
)
def test_package_imports(module_name: str) -> None:
    """Verify all repository packages import without errors."""
    mod = importlib.import_module(module_name)
    assert mod is not None


def test_configs_exist_and_parse() -> None:
    """Verify all YAML configs parse and define required keys."""
    configs_dir = REPO_ROOT / "configs"
    expected_configs = {
        "sft_lora.yaml": ["model_name_or_path", "lora_r", "lora_alpha"],
        "rest.yaml": ["num_iterations", "rollouts_per_task", "speedup_thresholds"],
        "multiturn_grpo.yaml": ["gamma_discount", "max_grad_norm", "group_size"],
        "contrastive_rl.yaml": [
            "enable_cross_trajectory_anchors",
            "min_speedup_margin_delta",
            "pairwise_margin_weight",
        ],
    }

    for config_filename, required_keys in expected_configs.items():
        config_path = configs_dir / config_filename
        assert config_path.is_file(), f"Missing config file: {config_filename}"
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), f"Config {config_filename} is not a valid dict"
        for key in required_keys:
            assert key in data, f"Key '{key}' missing from {config_filename}"


def test_documentation_files_exist_and_nonempty() -> None:
    """Verify core architecture, contracts, and guide docs are populated."""
    doc_paths = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs" / "ARCHITECTURE.md",
        REPO_ROOT / "docs" / "CONTRACTS.md",
        REPO_ROOT / "src" / "verifier" / "README.md",
        REPO_ROOT / "src" / "baselines" / "README.md",
        REPO_ROOT / "src" / "analytics" / "README.md",
        REPO_ROOT / "data" / "README.md",
        REPO_ROOT / "scripts" / "README.md",
    ]
    for path in doc_paths:
        assert path.is_file(), f"Document missing: {path}"
        assert path.stat().st_size > 50, f"Document is too short or empty: {path}"


def test_contracts_schema_and_mock_verifier() -> None:
    """Verify data contracts instantiate and mock verifier works offline."""
    from src.contracts import KernelEvalRequest, MockVerifier

    request = KernelEvalRequest(
        task_id="test_vector_add",
        code="__global__ void add() {}",
        baseline_code="__global__ void add_ref() {}",
    )
    verifier = MockVerifier(simulated_speedup=1.20)
    response = verifier.evaluate(request)

    assert response.task_id == "test_vector_add"
    assert response.compiled is True
    assert response.correct is True
    assert response.speedup == 1.20
