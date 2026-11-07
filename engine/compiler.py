import re
import uuid
from typing import Any, Optional, Tuple

# Compiles candidate CUDA kernels using PyTorch inline C++ extensions.
# Performs static inspection to enforce interface contracts and prevent framework shortcuts.
# Employs unique module naming per compilation to avoid symbol caching collisions.

FORBIDDEN_SHORTCUTS = [
    "torch::relu",
    "torch::sigmoid",
    "at::native",
    "torch::nn::functional",
    "torch::max(",
    "torch::sum(",
    "torch::mean(",
    "at::relu",
    "at::sigmoid",
]

def extract_code_block(raw_text: str) -> str:
    pattern = r"```(?:cpp|cuda|c\+\+)?\s*(.*?)\s*```"
    matches = re.findall(pattern, raw_text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return raw_text.strip()

def strip_comments(code: str) -> str:
    """Strip C/C++ style single-line and multi-line comments."""
    no_block = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    no_line = re.sub(r"//.*", "", no_block)
    return no_line

def run_static_checks(code: str) -> Tuple[bool, str]:
    if "__global__" not in code:
        return False, "Missing __global__ kernel declaration."
    if "run_cuda" not in code or "torch::Tensor" not in code:
        return False, "Missing required host entrypoint: torch::Tensor run_cuda(torch::Tensor x)."

    clean_code = strip_comments(code)
    for shortcut in FORBIDDEN_SHORTCUTS:
        if shortcut in clean_code:
            return False, f"Forbidden PyTorch shortcut detected: {shortcut}."
    return True, "Static inspection passed."

def compile_kernel(code_or_completion: str, verbose: bool = False) -> Tuple[Optional[Any], str]:
    code = extract_code_block(code_or_completion)
    passed_static, reason = run_static_checks(code)
    if not passed_static:
        return None, reason

    try:
        from torch.utils.cpp_extension import load_inline
    except ImportError:
        return None, "PyTorch is not available for compilation."

    # Prevent duplicate module definition if model output included PYBIND11_MODULE
    if "PYBIND11_MODULE" in code:
        code = re.sub(r"PYBIND11_MODULE\s*\([^)]*\)\s*\{[^}]*\}", "", code)

    module_name = f"cuda_kernel_{uuid.uuid4().hex[:12]}"
    cpp_source = """
    #include <torch/extension.h>
    torch::Tensor run_cuda(torch::Tensor x);
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("run_cuda", &run_cuda, "CUDA kernel runner");
    }
    """

    try:
        module = load_inline(
            name=module_name,
            cpp_sources=cpp_source,
            cuda_sources=code,
            functions=None,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=verbose,
        )
        return module, "Compilation successful."
    except Exception as e:
        error_msg = str(e)
        return None, f"Compilation failed: {error_msg[:300]}"
