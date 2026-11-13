import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from engine.utils import sha256_text

# Compiles candidate CUDA kernels using PyTorch inline C++ extensions.
# Static inspection is intentionally conservative because the model controls the code.

ALLOWED_TORCH_SYMBOLS = {
    "Tensor",
    "empty",
    "empty_like",
    "zeros",
    "ones",
    "full",
    "full_like",
    "kFloat",
    "kFloat32",
    "kFloat16",
    "kBFloat16",
    "kInt",
    "kInt32",
    "kLong",
    "kCUDA",
    "kCPU",
    "Device",
    "ScalarType",
    "TensorOptions",
    "from_blob",
    "cuda",
}

FORBIDDEN_MEMBER_METHODS = {
    "relu",
    "sigmoid",
    "exp",
    "log",
    "softmax",
    "log_softmax",
    "max",
    "min",
    "sum",
    "mean",
    "cumsum",
    "pow",
    "rsqrt",
    "sqrt",
    "norm",
    "avg_pool2d",
    "max_pool2d",
    "matmul",
    "mm",
    "bmm",
    "add",
    "mul",
    "div",
    "sub",
}

FORBIDDEN_SNIPPETS = [
    "at::native",
    "torch::nn::functional",
    "ATen/native",
    "cublas",
    "cudnn",
    "cudaMemcpy",
    "cudaMemset",
]


@dataclass
class CompileResult:
    module: Optional[Any]
    status: str
    message: str
    code: str
    code_hash: str
    module_name: str
    build_time_sec: float
    log: str = ""


def extract_code_block(raw_text: str) -> str:
    pattern = r"```(?:cpp|cuda|c\+\+)?\s*(.*?)\s*```"
    matches = re.findall(pattern, raw_text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return raw_text.strip()


def strip_comments_and_strings(code: str) -> str:
    no_block = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    no_line = re.sub(r"//.*", "", no_block)
    no_strings = re.sub(r'"(?:\\.|[^"\\])*"', '""', no_line)
    no_chars = re.sub(r"'(?:\\.|[^'\\])*'", "''", no_strings)
    return no_chars


def strip_comments(code: str) -> str:
    return strip_comments_and_strings(code)


def sanitize_source(code: str) -> str:
    code = re.sub(r"^[ \t]*#include[ \t]*(?:\r?\n|$)", "\n", code, flags=re.MULTILINE)
    code = re.sub(r"(?<=\.data_ptr)\(\)", "<float>()", code)
    if "PYBIND11_MODULE" in code:
        code = re.sub(r"PYBIND11_MODULE\s*\([^)]*\)\s*\{.*?\n\}", "", code, flags=re.DOTALL)
        code = re.sub(r"PYBIND11_MODULE\s*\([^)]*\)\s*\{[^}]*\}", "", code)
    return code


def run_static_checks(code: str) -> Tuple[bool, str]:
    if "__global__" not in code:
        return False, "Missing __global__ kernel declaration."
    if "run_cuda" not in code or "torch::Tensor" not in code:
        return False, "Missing required host entrypoint: torch::Tensor run_cuda(torch::Tensor x)."

    clean_code = strip_comments_and_strings(code)

    for snippet in FORBIDDEN_SNIPPETS:
        if snippet in clean_code:
            return False, f"Forbidden PyTorch shortcut detected: {snippet}."

    for match in re.finditer(r"\b(?:torch|at)::([A-Za-z_][A-Za-z0-9_]*)", clean_code):
        symbol = match.group(1)
        if symbol not in ALLOWED_TORCH_SYMBOLS:
            return False, f"Forbidden PyTorch shortcut detected: {match.group(0)}."

    for match in re.finditer(r"\.\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", clean_code):
        method = match.group(1)
        if method in FORBIDDEN_MEMBER_METHODS:
            return False, f"Forbidden PyTorch shortcut detected: .{method}()."

    if "cudaMalloc" in clean_code or "cudaFree" in clean_code:
        return False, "Forbidden manual CUDA allocation detected. Use torch::empty."

    return True, "Static inspection passed."


def compile_kernel_ex(
    code_or_completion: str,
    verbose: bool = False,
    extra_cuda_cflags: Optional[list] = None,
) -> CompileResult:
    started = time.perf_counter()
    code = sanitize_source(extract_code_block(code_or_completion))
    code_hash = sha256_text(code)
    module_name = f"cuda_kernel_{code_hash[:12]}_{uuid.uuid4().hex[:8]}"

    passed_static, reason = run_static_checks(code)
    if not passed_static:
        return CompileResult(
            module=None,
            status="static_failed",
            message=reason,
            code=code,
            code_hash=code_hash,
            module_name=module_name,
            build_time_sec=time.perf_counter() - started,
        )

    try:
        from torch.utils.cpp_extension import load_inline
    except ImportError:
        return CompileResult(
            module=None,
            status="runtime_unavailable",
            message="PyTorch is not available for compilation.",
            code=code,
            code_hash=code_hash,
            module_name=module_name,
            build_time_sec=time.perf_counter() - started,
        )

    cpp_source = """
    #include "torch/extension.h"
    torch::Tensor run_cuda(torch::Tensor x);
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("run_cuda", &run_cuda, "CUDA kernel runner");
    }
    """
    flags = ["-O3", "--use_fast_math"] + list(extra_cuda_cflags or [])

    try:
        module = load_inline(
            name=module_name,
            cpp_sources=cpp_source,
            cuda_sources=code,
            functions=None,
            extra_cuda_cflags=flags,
            verbose=verbose,
        )
        return CompileResult(
            module=module,
            status="compiled",
            message="Compilation successful.",
            code=code,
            code_hash=code_hash,
            module_name=module_name,
            build_time_sec=time.perf_counter() - started,
        )
    except Exception as exc:
        return CompileResult(
            module=None,
            status="compile_error",
            message=f"Compilation failed: {str(exc)[:500]}",
            code=code,
            code_hash=code_hash,
            module_name=module_name,
            build_time_sec=time.perf_counter() - started,
            log=str(exc)[:4000],
        )


def compile_kernel(code_or_completion: str, verbose: bool = False) -> Tuple[Optional[Any], str]:
    result = compile_kernel_ex(code_or_completion, verbose=verbose)
    return result.module, result.message
