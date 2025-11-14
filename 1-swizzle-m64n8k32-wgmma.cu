// TL+ {"platform": "h100"}
// TL+ {"header_files": ["tma-interface.cuh", "wgmma-interface.cuh"]}
// TL+ {"compile_flags": ["-lcuda"]}

#include <cuda.h>
#include <cuda_bf16.h>
#include <iostream>
#include <stdio.h>

#include "tma-interface.cuh"
#include "wgmma-interface.cuh"

typedef __nv_bfloat16 bf16;

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
// Part 0: 64B Swizzle WGGMA load for M = 64, N = 8, K = 32
////////////////////////////////////////////////////////////////////////////////
template <int TILE_M, int TILE_N, int TILE_K>
__global__ void swizzle_wgmma_m64n8k32(
    __grid_constant__ const CUtensorMap a,
    __grid_constant__ const CUtensorMap b,
    float *c) {
    extern __shared__ uint8_t shmem_raw[(TILE_M * TILE_K + TILE_K * TILE_N) * sizeof(bf16) + 128];
    
    uint64_t* bar_A = reinterpret_cast<uint64_t*>(shmem_raw);
    uint64_t* bar_B = reinterpret_cast<uint64_t*>(shmem_raw + 8);
    
    uintptr_t tensor_base_addr = reinterpret_cast<uintptr_t>(shmem_raw + 16);
    tensor_base_addr = (tensor_base_addr + 127) & ~127ULL;
    
    bf16* smem_a = reinterpret_cast<bf16*>(tensor_base_addr);
    bf16* smem_b = smem_a + TILE_M * TILE_K;
    
    int tid = threadIdx.x;
    int lane = tid % 32;

    if (tid == 0) {
        init_barrier(bar_A, 1);
        init_barrier(bar_B, 1);
    }
    __syncthreads();
    async_proxy_fence();

    if (tid == 0) {
        expect_bytes(bar_A, TILE_M * TILE_K * sizeof(bf16));
        expect_bytes(bar_B, TILE_N * TILE_K * sizeof(bf16));
        cp_async_bulk_tensor_2d_global_to_shared(smem_a, &a, 0, 0, bar_A);
        cp_async_bulk_tensor_2d_global_to_shared(smem_b, &b, 0, 0, bar_B);
        arrive(bar_A, 1);
        arrive(bar_B, 1);
    }

    // Wait for TMA to complete
    wait(bar_A, 0);
    wait(bar_B, 0);

    __syncthreads();
    warpgroup_arrive();
    
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    uint64_t desc_a1 = make_smem_desc<SWIZZLE_64B>(smem_a, 0, 512);
    uint64_t desc_b1 = make_smem_desc<SWIZZLE_64B>(smem_b, 0, 0);

    wgmma_n8<1, 1, 1, 0, 0>(desc_a1, desc_b1, acc);

    uint64_t desc_a2 = make_smem_desc<SWIZZLE_64B>(smem_a + 16, 0, 512);
    uint64_t desc_b2 = make_smem_desc<SWIZZLE_64B>(smem_b + 16, 0, 0);
    
    wgmma_n8<1, 1, 1, 0, 0>(desc_a2, desc_b2, acc);
    
    wgmma_commit();
    wgmma_wait<0>();

    for (int i = 0; i < 4; i++) {
        c[(tid % 32 ) / 4 + (tid / 32) * 16 + (tid % 4)*128 + (i%2)*64 + (i/2)*8] = acc[i];
    }
}

template <int TILE_M, int TILE_N, int TILE_K>
void launch_swizzle_wgmma_m64n8k32(bf16 *a, bf16 *b, float *c) {
    CUtensorMap CUtensorMapA;
    CUtensorMap CUtensorMapB;

    // A is 64x32
    const cuuint64_t global_dim_a[2] = {TILE_K, TILE_M};
    const cuuint64_t global_strides_a[2] = {TILE_K * sizeof(bf16)};
    const cuuint32_t box_dim_a[2] = {TILE_K, TILE_M};
    const cuuint32_t element_strides_a[2] = {1, 1};

    cuTensorMapEncodeTiled(
        &CUtensorMapA, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 
        2, 
        a, 
        global_dim_a,
        global_strides_a, 
        box_dim_a, 
        element_strides_a,
        CU_TENSOR_MAP_INTERLEAVE_NONE, 
        CU_TENSOR_MAP_SWIZZLE_64B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    // B is 32x8
    const cuuint64_t global_dim_b[2] = {TILE_K, TILE_N};
    const cuuint64_t global_strides_b[2] = {TILE_K * sizeof(bf16)};
    const cuuint32_t box_dim_b[2] = {TILE_K, TILE_N};
    const cuuint32_t element_strides_b[2] = {1, 1};

    cuTensorMapEncodeTiled(
        &CUtensorMapB, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 
        2, 
        b, 
        global_dim_b,
        global_strides_b, 
        box_dim_b, 
        element_strides_b,
        CU_TENSOR_MAP_INTERLEAVE_NONE, 
        CU_TENSOR_MAP_SWIZZLE_64B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    swizzle_wgmma_m64n8k32<TILE_M, TILE_N, TILE_K><<<1, 128>>>(CUtensorMapA, CUtensorMapB, c);
}
////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

int main() {
    const int M = 64;
    const int N = 8;
    const int K = 32;

    // Initialize source matrix on host
    bf16 *a = (bf16 *)malloc(M * K * sizeof(bf16));
    bf16 *b = (bf16 *)malloc(N * K * sizeof(bf16));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            a[i * K + j] = (i + j) / 10.0f;
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            b[i * K + j] = (i + j) / 10.0f;
        }
    }

    float *d_c;
    bf16 *d_a, *d_b;
    cudaMalloc(&d_a, M * K * sizeof(bf16));
    cudaMalloc(&d_b, N * K * sizeof(bf16));
    cudaMalloc(&d_c, M * N * sizeof(float));
    cudaMemcpy(d_a, a, M * K * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * K * sizeof(bf16), cudaMemcpyHostToDevice);

    // Compute CPU reference
    float *cpu_output = (float *)malloc(M * N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float temp = 0.0f;
            for (int k = 0; k < K; k++) {
                float a_row = (float)a[i * K + k];
                float a_col = (float)b[k + j * K];
                temp += a_row * a_col;
            }
            cpu_output[j * M + i] = temp;
        }
    }

    float *gpu_output = (float *)malloc(M * N * sizeof(float));
    for (int i = 0; i < M * N; i++) {
        gpu_output[i] = 0;
    }
    cudaMemcpy(d_c, gpu_output, M * N * sizeof(float), cudaMemcpyHostToDevice);

    printf("\n\nRunning Swizzle WGMMA M=64, N=8, K-32...\n\n");
    launch_swizzle_wgmma_m64n8k32<M, N, K>(d_a, d_b, d_c);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    cudaMemcpy(gpu_output, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // check results
    bool correct = true;
    for (int idx = 0; idx < M * N; idx++) {
        if (fabs(cpu_output[idx] - gpu_output[idx]) > 0.01f) {
            correct = false;
            int j = idx / M;
            int i = idx % M;
            printf(
                "\nFirst mismatch at (%d, %d): CPU=%.0f, GPU=%.0f\n",
                i,
                j,
                cpu_output[idx],
                gpu_output[idx]);
            break;
        }
    }

    printf("%s output!\n\n\n", correct ? "Correct" : "Incorrect");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);

    return 0;
}
