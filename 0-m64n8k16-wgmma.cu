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
// Part 0: No Swizzle WGGMA load for M = 64, N = 8, K = 16
////////////////////////////////////////////////////////////////////////////////

template <int TILE_M, int TILE_N, int TILE_K>
__global__ void wgmma_m64n8k16(bf16 *a, bf16 *b, float *c) {

    __shared__ bf16 smem_a[TILE_M * TILE_K];  // 64 x 16
    __shared__ bf16 smem_b[TILE_K * TILE_N]; // 16 x 8

    int tid = threadIdx.x;

    for (int i = tid; i < TILE_M * TILE_K; i += blockDim.x) {
        smem_a[i] = a[i];
    }

    for (int i = tid; i < TILE_K * TILE_N; i += blockDim.x) {
        smem_b[i] = b[(i % TILE_K) * TILE_N + (i / TILE_K)];
        //smem_b[i] = b[i];
    }

    __syncthreads();

    uint64_t a_desc = make_smem_desc<NO_SWIZZLE>(smem_a, 128, 256);
    uint64_t b_desc = make_smem_desc<NO_SWIZZLE>(smem_b, 128, 128);

    float acc[4] = {0.0f};

    warpgroup_arrive();

    wgmma_n8<1, 1, 1, 0, 0>(a_desc, b_desc, acc);

    wgmma_commit();
    wgmma_wait<0>();

    // need to write acc valuies back in to c
    for (int i = 0; i < 4; i++) {
        c[(tid % 32 ) / 4 + (tid / 32) * 16 + (tid % 4)*128 + (i%2)*64 + (i/2)*8] = acc[i];
    }


    __syncthreads();

    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 8; j++) {
            if (tid == 0) { 
                printf("c[%d, %d] = %.0f\n", i, j, c[j * 64 + i]);
            }
        }
    }

} 

template <int TILE_M, int TILE_N, int TILE_K>
void launch_wgmma_m64n8k16(bf16 *a, bf16 *b, float *c) {
    wgmma_m64n8k16<TILE_M, TILE_N, TILE_K><<<1, 128>>>(a, b, c);

}


////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

int main() {
    const int M = 64;
    const int N = 8;
    const int K = 16;

    // Initialize source matrix on host
    bf16 *a = (bf16 *)malloc(M * K * sizeof(bf16));
    bf16 *b = (bf16 *)malloc(N * K * sizeof(bf16));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            a[i * K + j] = i + j;
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            b[j * N + i] = i + j;
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

    printf("\n\nRunning No Swizzle WGMMA M=%d, N=%d, K=%d...\n\n", M, N, K);
    launch_wgmma_m64n8k16<M, N, K>(d_a, d_b, d_c);
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