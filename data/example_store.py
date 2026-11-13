from typing import Any, Dict, List

# Curated repository of teacher CUDA kernels with measured baseline speedup scores.
# Provides reference kernels extracted from CUDA-L1 for contrastive prompt generation.
# Tasks with single teachers are handled gracefully by returning all available variants.

TEACHER_EXAMPLES: Dict[str, List[Dict[str, Any]]] = {
    "swish": [
        {
            "name": "basic_elementwise_swish",
            "speedup": 1.15,
            "description": "Standard elementwise grid mapping using fast exp intrinsics.",
            "code": """#include "torch/extension.h"
#include "cuda_runtime.h"

__global__ void swish_kernel_basic(
    const float* __restrict__ input, float* __restrict__ output, int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float x = input[idx];
        output[idx] = x / (1.0f + __expf(-x));
    }
}

torch::Tensor run_cuda(torch::Tensor x) {
    auto output = torch::empty_like(x);
    int total_elements = x.numel();
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;
    const float* in_ptr = (const float*)x.data_ptr();
    float* out_ptr = (float*)output.data_ptr();
    swish_kernel_basic<<<grid_size, block_size>>>(
        in_ptr, out_ptr, total_elements
    );
    return output;
}""",
        },
        {
            "name": "vectorized_float4_swish",
            "speedup": 1.28,
            "description": "Vectorized 128-bit memory loads and stores with float4 registers.",
            "code": """#include "torch/extension.h"
#include "cuda_runtime.h"

__global__ void swish_kernel_vectorized(
    const float4* __restrict__ input, float4* __restrict__ output, int total_vecs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_vecs) {
        float4 in = input[idx];
        float4 out;
        out.x = in.x / (1.0f + __expf(-in.x));
        out.y = in.y / (1.0f + __expf(-in.y));
        out.z = in.z / (1.0f + __expf(-in.z));
        out.w = in.w / (1.0f + __expf(-in.w));
        output[idx] = out;
    }
}

torch::Tensor run_cuda(torch::Tensor x) {
    auto output = torch::empty_like(x);
    int total_elements = x.numel();
    int total_vecs = total_elements / 4;
    const int block_size = 256;
    const int grid_size = (total_vecs + block_size - 1) / block_size;
    const float4* in_ptr = (const float4*)x.data_ptr();
    float4* out_ptr = (float4*)output.data_ptr();
    swish_kernel_vectorized<<<grid_size, block_size>>>(
        in_ptr, out_ptr, total_vecs
    );
    return output;
}""",
        },
    ],
    "column_max": [
        {
            "name": "linear_column_scan",
            "speedup": 1.10,
            "description": "Per column thread mapping with contiguous memory striding.",
            "code": """#include "torch/extension.h"
#include "cuda_runtime.h"

__global__ void column_max_linear_kernel(
    const float* __restrict__ input, float* __restrict__ output, int rows, int cols
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        float max_val = input[col];
        for (int r = 1; r < rows; ++r) {
            float val = input[r * cols + col];
            if (val > max_val) {
                max_val = val;
            }
        }
        output[col] = max_val;
    }
}

torch::Tensor run_cuda(torch::Tensor x) {
    int rows = x.size(0);
    int cols = x.size(1);
    auto output = torch::empty({cols}, x.options());
    const int block_size = 256;
    const int grid_size = (cols + block_size - 1) / block_size;
    const float* in_ptr = (const float*)x.data_ptr();
    float* out_ptr = (float*)output.data_ptr();
    column_max_linear_kernel<<<grid_size, block_size>>>(
        in_ptr, out_ptr, rows, cols
    );
    return output;
}""",
        },
        {
            "name": "unrolled_column_max",
            "speedup": 1.35,
            "description": "Unrolled column reduction loop using register caching.",
            "code": """#include "torch/extension.h"
#include "cuda_runtime.h"

__global__ void column_max_unrolled_kernel(
    const float* __restrict__ input, float* __restrict__ output, int rows, int cols
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        float max_val = input[col];
        #pragma unroll 4
        for (int r = 1; r < rows; ++r) {
            float val = input[r * cols + col];
            max_val = fmaxf(max_val, val);
        }
        output[col] = max_val;
    }
}

torch::Tensor run_cuda(torch::Tensor x) {
    int rows = x.size(0);
    int cols = x.size(1);
    auto output = torch::empty({cols}, x.options());
    const int block_size = 128;
    const int grid_size = (cols + block_size - 1) / block_size;
    const float* in_ptr = (const float*)x.data_ptr();
    float* out_ptr = (float*)output.data_ptr();
    column_max_unrolled_kernel<<<grid_size, block_size>>>(
        in_ptr, out_ptr, rows, cols
    );
    return output;
}""",
        },
    ],
    "channel_rmsnorm": [
        {
            "name": "warp_reduced_rmsnorm",
            "speedup": 1.20,
            "description": "Warp shuffle tree reduction along the trailing channel dimension.",
            "code": """#include "torch/extension.h"
#include "cuda_runtime.h"

__global__ void rmsnorm_warp_kernel(
    const float* __restrict__ input, float* __restrict__ output, int rows, int cols, float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    float sum_sq = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        float v = input[row * cols + col];
        sum_sq += v * v;
    }
    __shared__ float sdata[32];
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    int lane = tid % 32;
    int wid = tid / 32;
    if (lane == 0) {
        sdata[wid] = sum_sq;
    }
    __syncthreads();
    float block_sum = 0.0f;
    if (tid < 32) {
        block_sum = (tid < (blockDim.x / 32)) ? sdata[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
    }
    __shared__ float scale;
    if (tid == 0) {
        float mean_sq = block_sum / (float)cols;
        scale = rsqrtf(mean_sq + eps);
    }
    __syncthreads();
    for (int col = tid; col < cols; col += blockDim.x) {
        output[row * cols + col] = input[row * cols + col] * scale;
    }
}

torch::Tensor run_cuda(torch::Tensor x) {
    auto output = torch::empty_like(x);
    int cols = x.size(-1);
    int rows = x.numel() / cols;
    const int block_size = 256;
    const float* in_ptr = (const float*)x.data_ptr();
    float* out_ptr = (float*)output.data_ptr();
    rmsnorm_warp_kernel<<<rows, block_size>>>(
        in_ptr, out_ptr, rows, cols, 1e-5f
    );
    return output;
}""",
        }
    ],
    "average_pooling_2d": [
        {
            "name": "block_tiled_avg_pool2d",
            "speedup": 1.18,
            "verified": False,
            "description": "One thread per output pixel with vectorized four-value reduction.",
            "code": """#include "torch/extension.h"
#include "cuda_runtime.h"

__global__ void avg_pool2d_kernel(
    const float* __restrict__ input, float* __restrict__ output,
    int n, int c, int h, int w, int oh, int ow
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * c * oh * ow;
    if (idx >= total) {
        return;
    }
    int ow_idx = idx % ow;
    int oh_idx = (idx / ow) % oh;
    int c_idx = (idx / (ow * oh)) % c;
    int n_idx = idx / (ow * oh * c);
    int ih = oh_idx * 2;
    int iw = ow_idx * 2;
    const float* base = input + ((n_idx * c + c_idx) * h + ih) * w + iw;
    float sum = base[0] + base[1] + base[w] + base[w + 1];
    output[idx] = sum * 0.25f;
}

torch::Tensor run_cuda(torch::Tensor x) {
    int n = x.size(0);
    int c = x.size(1);
    int h = x.size(2);
    int w = x.size(3);
    int oh = h / 2;
    int ow = w / 2;
    auto output = torch::empty({n, c, oh, ow}, x.options());
    int total = n * c * oh * ow;
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;
    const float* in_ptr = (const float*)x.data_ptr();
    float* out_ptr = (float*)output.data_ptr();
    avg_pool2d_kernel<<<grid_size, block_size>>>(
        in_ptr, out_ptr, n, c, h, w, oh, ow
    );
    return output;
}""",
        },
    ],
    "softmax": [
        {
            "name": "block_reduced_softmax",
            "speedup": 1.22,
            "verified": False,
            "description": "One block per row with shared memory max and sum reductions.",
            "code": """#include "torch/extension.h"
#include "cuda_runtime.h"
#include <float.h>

__global__ void softmax_kernel(
    const float* __restrict__ input, float* __restrict__ output, int rows, int cols
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* in_row = input + row * cols;
    float* out_row = output + row * cols;
    float max_val = -FLT_MAX;
    for (int i = tid; i < cols; i += blockDim.x) {
        max_val = fmaxf(max_val, in_row[i]);
    }
    __shared__ float sdata[256];
    sdata[tid] = max_val;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }
    max_val = sdata[0];
    float sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        sum += __expf(in_row[i] - max_val);
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    float inv_sum = 1.0f / sdata[0];
    for (int i = tid; i < cols; i += blockDim.x) {
        out_row[i] = __expf(in_row[i] - max_val) * inv_sum;
    }
}

torch::Tensor run_cuda(torch::Tensor x) {
    int cols = x.size(-1);
    int rows = x.numel() / cols;
    auto output = torch::empty_like(x);
    const float* in_ptr = (const float*)x.data_ptr();
    float* out_ptr = (float*)output.data_ptr();
    softmax_kernel<<<rows, 256>>>(in_ptr, out_ptr, rows, cols);
    return output;
}""",
        },
    ],
    "cumsum": [
        {
            "name": "warp_chunk_scan",
            "speedup": 1.15,
            "verified": False,
            "description": (
                "One warp per row with chunked local scans and a warp-level prefix scan."
            ),
            "code": """#include "torch/extension.h"
#include "cuda_runtime.h"

__global__ void cumsum_kernel(
    const float* __restrict__ input, float* __restrict__ output, int rows, int cols
) {
    int row = blockIdx.x;
    int lane = threadIdx.x;
    const float* in_row = input + row * cols;
    float* out_row = output + row * cols;
    int chunk = (cols + 31) / 32;
    int start = lane * chunk;
    int end = start + chunk;
    if (end > cols) {
        end = cols;
    }
    float local_sum = 0.0f;
    for (int i = start; i < end; ++i) {
        local_sum += in_row[i];
        out_row[i] = local_sum;
    }
    float scan = local_sum;
    for (int offset = 1; offset < 32; offset <<= 1) {
        float value = __shfl_up_sync(0xffffffff, scan, offset);
        if (lane >= offset) {
            scan += value;
        }
    }
    float prefix = scan - local_sum;
    for (int i = start; i < end; ++i) {
        out_row[i] += prefix;
    }
}

torch::Tensor run_cuda(torch::Tensor x) {
    int cols = x.size(-1);
    int rows = x.numel() / cols;
    auto output = torch::empty_like(x);
    const float* in_ptr = (const float*)x.data_ptr();
    float* out_ptr = (float*)output.data_ptr();
    cumsum_kernel<<<rows, 32>>>(in_ptr, out_ptr, rows, cols);
    return output;
}""",
        },
    ],
}


_MEASUREMENT_CACHE: Dict[str, Any] = {}


def _load_measurements() -> Dict[str, float]:
    import json
    import os
    from pathlib import Path

    default_path = Path(__file__).with_name("teacher_measurements.json")
    path = Path(os.environ.get("TEACHER_MEASUREMENTS_PATH", str(default_path)))
    if not path.exists():
        _MEASUREMENT_CACHE.pop(str(path), None)
        return {}

    stat = path.stat()
    cached = _MEASUREMENT_CACHE.get(str(path))
    if cached and cached["mtime"] == stat.st_mtime and cached["size"] == stat.st_size:
        return cached["data"]

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    measurements = {
        f"{task_id}:{entry['name']}": float(entry["speedup"])
        for task_id, entries in payload.items()
        for entry in entries
    }
    _MEASUREMENT_CACHE[str(path)] = {
        "mtime": stat.st_mtime,
        "size": stat.st_size,
        "data": measurements,
    }
    return measurements


def get_teacher_examples(task_id: str) -> List[Dict[str, Any]]:
    teachers = [dict(teacher) for teacher in TEACHER_EXAMPLES.get(task_id, [])]
    measurements = _load_measurements()
    for teacher in teachers:
        measured = measurements.get(f"{task_id}:{teacher['name']}")
        if measured is not None:
            teacher["speedup"] = measured
            teacher["verified"] = True
    return teachers
