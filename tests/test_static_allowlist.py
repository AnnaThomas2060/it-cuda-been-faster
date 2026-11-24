"""Static checks must allow host-side setup helpers but still block compute shortcuts."""

import pytest

from engine.compiler import run_static_checks

KERNEL = """
__global__ void k(const float* x, float* y, int n) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i];
}}
torch::Tensor run_cuda(torch::Tensor x) {{
    {host}
    return y;
}}
"""


@pytest.mark.parametrize("host", [
    "auto y = torch::empty({x.numel()}, torch::dtype(torch::kFloat32).device(x.device()));",
    "auto y = torch::zeros_like(x);",
    "auto y = torch::empty_strided(x.sizes(), x.strides(), x.options());",
    "at::DeviceGuard guard(x.device()); auto y = torch::empty_like(x);",
    "at::IntArrayRef s = x.sizes(); auto y = torch::empty(s, x.options());",
    "auto y = torch::empty_like(x, torch::dtype(torch::kHalf));",
])
def test_host_setup_helpers_are_allowed(host):
    ok, reason = run_static_checks(KERNEL.format(host=host))
    assert ok, reason


@pytest.mark.parametrize("host", [
    "auto y = torch::sigmoid(x);",
    "auto y = at::relu(x);",
    "auto y = x.sum();",
])
def test_compute_shortcuts_are_still_blocked(host):
    ok, _ = run_static_checks(KERNEL.format(host=host))
    assert not ok
