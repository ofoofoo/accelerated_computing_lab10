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
__device__ void swizzle_64B_chunk(bf16* shared_data, int lane) {
    bf16 my_element = shared_data[lane];
    
    int my_chunk = lane >> 3;
    int xor_bits = (lane >> 1) & 3;
    int new_chunk = my_chunk ^ xor_bits;
    int offset_in_chunk = lane & 7;
    int new_idx = (new_chunk << 3) | offset_in_chunk;
    
    bf16 swizzled_element = __shfl_sync(0xffffffff, my_element, new_idx);
    
    __syncwarp();
    shared_data[lane] = swizzled_element;
    __syncwarp();
}
////////////////////////////////////////////////////////////////////////////////
// 64-byte Swizzled WGMMA for M=64, N=8, K=32
////////////////////////////////////////////////////////////////////////////////

template <int TILE_M, int TILE_N, int TILE_K>
__global__ void swizzle_wgmma_m64n8k32(bf16 *a, bf16 *b, float *c) {

    __shared__ bf16 smem_a[TILE_M * TILE_K];  // 64 x 32
    __shared__ bf16 smem_b[TILE_K * TILE_N];  // 32 x 8

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;

    // ============ LOAD A (row-major) ============
    for (int i = tid; i < TILE_M * TILE_K; i += blockDim.x) {
        smem_a[i] = a[i];
    }

    // ============ LOAD B (NO TRANSPOSE - assume already column-major) ============
    for (int i = tid; i < TILE_K * TILE_N; i += blockDim.x) {
        smem_b[i] = b[i];
    }

    __syncthreads();

    // ============ APPLY 64-BYTE SWIZZLE ============
    
    // Swizzle A: each row is 32 elements = 64 bytes
    for (int row = warp_id; row < TILE_M; row += 4) {
        swizzle_64B_chunk(smem_a + row * TILE_K, lane);
    }

    __syncthreads();

    // Swizzle B: each column is 32 elements = 64 bytes  
    for (int col = warp_id; col < TILE_N; col += 4) {
        swizzle_64B_chunk(smem_b + col * TILE_K, lane);
    }

    __syncthreads();
    
    
    warpgroup_arrive();

    // ============ CREATE DESCRIPTORS ============
    // A: M=64, each core matrix is 8x32 elements = 8x64 bytes
    // SBO = stride to next core matrix in M = 8 rows * 64 bytes = 512
    uint64_t a_desc_base = make_smem_desc<SWIZZLE_64B>(smem_a, 1, 512);
    
    // B: N=8, only 1 core matrix in N dimension
    // SBO = 0 (no stride needed)
    uint64_t b_desc_base = make_smem_desc<SWIZZLE_64B>(smem_b, 1, 0);

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // ============ FIRST WGMMA: K = [0, 16) ============
    wgmma_n8<1, 1, 1, 0, 0>(a_desc_base, b_desc_base, acc);
    wgmma_commit();
    wgmma_wait<0>();  // Wait for first to complete

    // ============ SECOND WGMMA: K = [16, 32) ============
    // Offset by 16 elements = 32 bytes
    uint64_t a_desc_k16 = make_smem_desc<SWIZZLE_64B>(smem_a + 16, 1, 512);
    uint64_t b_desc_k16 = make_smem_desc<SWIZZLE_64B>(smem_b + 16, 1, 0);
    
    wgmma_n8<1, 1, 1, 0, 0>(a_desc_k16, b_desc_k16, acc);
    wgmma_commit();
    wgmma_wait<0>();  // Wait for second to complete

    // ============ WRITEBACK ============
    int m0 = 16 * warp_id + (lane / 4);
    int m1 = m0 + 8;
    int n0 = 2 * (lane % 4);
    int n1 = n0 + 1;

    c[n0 * 64 + m0] = acc[0];
    c[n1 * 64 + m0] = acc[1];
    c[n0 * 64 + m1] = acc[2];
    c[n1 * 64 + m1] = acc[3];

    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 8; j++) {
            if (tid == 0) { 
                printf("cpu_output[%d, %d] = %.0f\n", i, j, c[j * 64 + i]);
            }
        }
    }
}


template <int TILE_M, int TILE_N, int TILE_K>
void launch_swizzle_wgmma_m64n8k32(bf16 *a, bf16 *b, float *c) {
    swizzle_wgmma_m64n8k32<TILE_M, TILE_N, TILE_K><<<1, 128>>>(a, b, c);
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
            b[j * N + i] = (i + j) / 10.0f;
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

    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 8; j++) {
            printf("cpu_output[%d, %d] = %.0f\n", i, j, cpu_output[j * 64 + i]);
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