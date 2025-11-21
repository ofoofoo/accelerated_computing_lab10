// TL+ {"platform": "h100"}
// TL+ {"header_files": ["tma-interface.cuh", "wgmma-interface.cuh"]}
// TL+ {"compile_flags": ["-lcuda", "-lcublas"]}

#include <algorithm>
#include <cublas_v2.h>
#include <iostream>
#include <random>
#include <vector>
#include "tma-interface.cuh"
#include "wgmma-interface.cuh"

typedef __nv_bfloat16 bf16;

////////////////////////////////////////////////////////////////////////////////
// Part 1: Matrix Multiplication for M = 8192, N = 8192, K = 8192
////////////////////////////////////////////////////////////////////////////////
#define THREADS_PER_BLOCK 128
#define TILE_M 64
#define TILE_N 256
#define TILE_K 64

__global__ void h100_matmul(
    __grid_constant__ const CUtensorMap a_map,
    __grid_constant__ const CUtensorMap b_map,
    bf16 *c
) {
    extern __shared__ uint8_t shmem[];

    uint64_t* bar_A = reinterpret_cast<uint64_t*>(shmem);
    uint64_t* bar_B = reinterpret_cast<uint64_t*>(shmem + 8);
    
    uintptr_t tensor_base = reinterpret_cast<uintptr_t>(shmem + 16);
    tensor_base = (tensor_base + 127) & ~127ULL;
    
    bf16* smem_A = reinterpret_cast<bf16*>(tensor_base);
    bf16* smem_B = smem_A + TILE_M * TILE_K;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int block_m = blockIdx.y;
    int block_n = blockIdx.x;

    if (tid == 0) {
        init_barrier(bar_A, 1);
        init_barrier(bar_B, 1);
    }
    __syncthreads();
    async_proxy_fence();

    float acc[16][8];
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 8; j++) {
            acc[i][j] = 0.0f;
        }
    }

    int num_k_tiles = 8192 / TILE_K;
    int phase_A = 0;
    int phase_B = 0;
    
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        if (tid == 0) {
            expect_bytes(bar_A, TILE_M * TILE_K * sizeof(bf16));
            expect_bytes(bar_B, TILE_K * TILE_N * sizeof(bf16));

            int coord_k = k_tile * TILE_K;
            int coord_m = block_m * TILE_M;
            int coord_n = block_n * TILE_N;
            
            // A is K-major (K, M), so coordinates are (k_tile, block_m)
            cp_async_bulk_tensor_2d_global_to_shared(
                smem_A, &a_map, coord_k, coord_m, bar_A
            );
            
            // B is K-major (K, N), so coordinates are (k_tile, block_n)
            cp_async_bulk_tensor_2d_global_to_shared(
                smem_B, &b_map, coord_k, coord_n, bar_B
            );
            
            arrive(bar_A, 1);
            arrive(bar_B, 1);
        }

        wait(bar_A, phase_A);
        wait(bar_B, phase_B);
        
        phase_A ^= 1;
        phase_B ^= 1;

        __syncthreads();
        warpgroup_arrive();

        uint64_t desc_a = make_smem_desc<SWIZZLE_128B>(smem_A, 0, 512);
        uint64_t desc_b = make_smem_desc<SWIZZLE_128B>(smem_B, 0, 0);

        wgmma_n256<1, 1, 1, 0, 0>(desc_a, desc_b, acc);

        uint64_t desc_a2 = make_smem_desc<SWIZZLE_128B>(smem_A + 16, 0, 512);
        uint64_t desc_b2 = make_smem_desc<SWIZZLE_128B>(smem_B + 16, 0, 0);
        wgmma_n256<1, 1, 1, 0, 0>(desc_a2, desc_b2, acc);

        uint64_t desc_a3 = make_smem_desc<SWIZZLE_128B>(smem_A + 32, 0, 512);
        uint64_t desc_b3 = make_smem_desc<SWIZZLE_128B>(smem_B + 32, 0, 0);
        wgmma_n256<1, 1, 1, 0, 0>(desc_a3, desc_b3, acc);

        uint64_t desc_a4 = make_smem_desc<SWIZZLE_128B>(smem_A + 48, 0, 512);
        uint64_t desc_b4 = make_smem_desc<SWIZZLE_128B>(smem_B + 48, 0, 0);
        wgmma_n256<1, 1, 1, 0, 0>(desc_a4, desc_b4, acc);

        wgmma_commit();
        wgmma_wait<0>();

        __syncthreads();
    }

    // writeback
// writeback - convert float to bf16
for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 8; j++) {
        int local_row = (tid % 4) * 16 + i;
        int local_col = (tid / 4) * 8 + j;
        
        int global_m = block_m * TILE_M + local_row;
        int global_n = block_n * TILE_N + local_col;
        
        // Convert float to bf16 before writing
        c[global_m * 8192 + global_n] = __float2bfloat16(acc[i][j]);
    }
}
}

void launch_h100_matmul(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C) {
    CUtensorMap a_map;
    CUtensorMap b_map;

    const cuuint64_t global_dim_a[2] = {(cuuint64_t)K, (cuuint64_t)M};
    const cuuint64_t global_strides_a[1] = {(cuuint64_t)(K * sizeof(bf16))};
    const cuuint32_t box_dim_a[2] = {TILE_K, TILE_M};
    const cuuint32_t element_strides_a[2] = {1, 1};

    cuTensorMapEncodeTiled(
        &a_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        A,
        global_dim_a,
        global_strides_a,
        box_dim_a,
        element_strides_a,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    const cuuint64_t global_dim_b[2] = {(cuuint64_t)K, (cuuint64_t)N};
    const cuuint64_t global_strides_b[1] = {(cuuint64_t)(K * sizeof(bf16))};
    const cuuint32_t box_dim_b[2] = {TILE_K, TILE_N};
    const cuuint32_t element_strides_b[2] = {1, 1};

    cuTensorMapEncodeTiled(
        &b_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        B,
        global_dim_b,
        global_strides_b,
        box_dim_b,
        element_strides_b,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    size_t shared_bytes = (TILE_M * TILE_K + TILE_K * TILE_N) * sizeof(bf16) + 16 + 128;

    int tile_m = (M + TILE_M - 1) / TILE_M;
    int tile_n = (N + TILE_N - 1) / TILE_N;
    dim3 grid(tile_n, tile_m);

    CUDA_CHECK(cudaFuncSetAttribute(
        h100_matmul,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        96000));
    
    h100_matmul<<<grid, THREADS_PER_BLOCK, shared_bytes>>>(
        a_map,
        b_map,
        // reinterpret_cast<float *>(C)
        C
    );
}

// each block does 64x256 output tile, similar to lab 4
// within each block: there are 2 WARPGROUPS: 1 producer 1 consudmer, walk along k dimension
// write into registers to accumulate results accordingly
// use write TMA to write back results to global memory
// extend to 1 producer 2 consumer later

// 1 producer 2 consumers: need ot make sure that when a consumer arrives at a point when TMA is ready, 
// swizzle with 128B needed: adds like 100 TFLOPs of gain, use 256 WGMMA

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

static constexpr size_t kNumOfWarmupIterations = 2;
static constexpr size_t kNumOfOuterIterations = 1;
static constexpr size_t kNumOfInnerIterations = 10;


#define BENCHPRESS(func, flops, ...)                                           \
    do {                                                                       \
        std::cout << "Running " << #func << " ...\n";                          \
        for (size_t i = 0; i < kNumOfWarmupIterations; ++i) {                  \
            func(__VA_ARGS__);                                                 \
        }                                                                      \
        cudaDeviceSynchronize();                                               \
        std::vector<float> times(kNumOfOuterIterations);                       \
        cudaEvent_t start, stop;                                               \
        cudaEventCreate(&start);                                               \
        cudaEventCreate(&stop);                                                \
        for (size_t i = 0; i < kNumOfOuterIterations; ++i) {                   \
            cudaEventRecord(start);                                            \
            for (size_t j = 0; j < kNumOfInnerIterations; ++j) {               \
                func(__VA_ARGS__);                                             \
            }                                                                  \
            cudaEventRecord(stop);                                             \
            cudaEventSynchronize(stop);                                        \
            float elapsed_time;                                                \
            cudaEventElapsedTime(&elapsed_time, start, stop);                  \
            times[i] = elapsed_time / kNumOfInnerIterations;                   \
        }                                                                      \
        cudaEventDestroy(start);                                               \
        cudaEventDestroy(stop);                                                \
        std::sort(times.begin(), times.end());                                 \
        float best_time_ms = times[0];                                         \
        float tflops = (flops * 1e-9) / best_time_ms;                          \
        std::cout << "  Runtime: " << best_time_ms << " ms" << std::endl;      \
        std::cout << "  TFLOP/s: " << tflops << std::endl;                     \
    } while (0)

void runCublasRef(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C) {
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    float alpha = 1, beta = 0;
    cublasStatus_t status =
        cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha,
                     A, CUDA_R_16BF, K, B, CUDA_R_16BF, K, &beta, C,
                     CUDA_R_16BF, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "CUBLAS error: " << status << std::endl;
        exit(1);
    }
}

void init_matrix(bf16 *mat, int N) {
    std::default_random_engine generator(0);
    std::normal_distribution<float> distribution(0, 1);
    for (int i = 0; i < N; i++) {
        mat[i] = distribution(generator);
    }
}

bool check_correctness(bf16 *ref, bf16 *test, int N, float tolerance = 0.1f) {
    int mismatches = 0;
    int total = N;
    for (int i = 0; i < N; i++) {
        float ref_val = __bfloat162float(ref[i]);
        float test_val = __bfloat162float(test[i]);
        float diff = std::abs(ref_val - test_val);
        if (diff > tolerance) {
            if (mismatches < 10) { // Print first 10 mismatches
                std::cout << "  Mismatch at index " << i << ": ref=" << ref_val
                          << ", test=" << test_val << ", diff=" << diff
                          << std::endl;
            }
            mismatches++;
        }
    }
    std::cout << "Total mismatches: " << mismatches << " / " << total << " ("
              << (100.0 * mismatches / total) << "%)" << std::endl;
    return mismatches == 0;
}

int main() {

    const int M = 8192, N = 8192, K = 8192;

    bf16 *A = (bf16 *)malloc(sizeof(bf16) * M * K);
    bf16 *B = (bf16 *)malloc(sizeof(bf16) * K * N);
    bf16 *C = (bf16 *)malloc(sizeof(bf16) * M * N);

    init_matrix(A, M * K);
    init_matrix(B, K * N);
    memset(C, 0, sizeof(bf16) * M * N);

    bf16 *dA;
    bf16 *dB;
    bf16 *dC;
    bf16 *dCublas;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(bf16) * M * K));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(bf16) * K * N));
    CUDA_CHECK(cudaMalloc(&dC, sizeof(bf16) * M * N));
    CUDA_CHECK(cudaMalloc(&dCublas, sizeof(bf16) * M * N));

    CUDA_CHECK(cudaMemcpy(dA, A, sizeof(bf16) * M * K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B, sizeof(bf16) * K * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, C, sizeof(bf16) * M * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(dCublas, C, sizeof(bf16) * M * N, cudaMemcpyHostToDevice));

    std::cout << "M = " << M << ", N = " << N << ", K = " << K << std::endl;

    bf16 *hCublas = (bf16 *)malloc(sizeof(bf16) * M * N);
    bf16 *hOurs = (bf16 *)malloc(sizeof(bf16) * M * N);

    runCublasRef(M, N, K, dA, dB, dCublas);
    launch_h100_matmul(M, N, K, dA, dB, dC);

    CUDA_CHECK(cudaMemcpy(hCublas, dCublas, sizeof(bf16) * M * N,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(
        cudaMemcpy(hOurs, dC, sizeof(bf16) * M * N, cudaMemcpyDeviceToHost));

    bool correct = check_correctness(hCublas, hOurs, M * N, 0.01f);
    printf("%s output!\n\n\n", correct ? "Correct" : "Incorrect");

    long flops = 2LL * M * N * K;
    BENCHPRESS(runCublasRef, flops, M, N, K, dA, dB, dCublas);

    BENCHPRESS(launch_h100_matmul, flops, M, N, K, dA, dB, dC);

    free(hCublas);
    free(hOurs);

    return 0;
}
