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
            "code": """#include <torch/extension.h>
#include <cuda_runtime.h>

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
    swish_kernel_basic<<<grid_size, block_size>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), total_elements
    );
    return output;
}""",
        },
        {
            "name": "vectorized_float4_swish",
            "speedup": 1.28,
            "description": "Vectorized 128-bit memory loads and stores with float4 registers.",
            "code": """#include <torch/extension.h>
#include <cuda_runtime.h>

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
    swish_kernel_vectorized<<<grid_size, block_size>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<float4*>(output.data_ptr<float>()),
        total_vecs
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
            "code": """#include <torch/extension.h>
#include <cuda_runtime.h>

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
    column_max_linear_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), rows, cols
    );
    return output;
}""",
        },
        {
            "name": "unrolled_column_max",
            "speedup": 1.35,
            "description": "Unrolled column reduction loop using register caching.",
            "code": """#include <torch/extension.h>
#include <cuda_runtime.h>

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
    column_max_unrolled_kernel<<<grid_size, block_size>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), rows, cols
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
            "code": """#include <torch/extension.h>
#include <cuda_runtime.h>

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
    rmsnorm_warp_kernel<<<rows, block_size>>>(
        x.data_ptr<float>(), output.data_ptr<float>(), rows, cols, 1e-5f
    );
    return output;
}""",
        }
    ],
}

def get_teacher_examples(task_id: str) -> List[Dict[str, Any]]:
    return TEACHER_EXAMPLES.get(task_id, [])
