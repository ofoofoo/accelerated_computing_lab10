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
#define TILE_M 128
#define TILE_N 256
#define TILE_K 64
#define MMA_TILE_K 16
#define MMA_TILE_M 64

#define TILE_SZ_BYTES ((TILE_M * TILE_K) + (TILE_K * TILE_N)) * sizeof(bf16)
#define OUT_TILE_SZ_BYTES (TILE_M * TILE_N * sizeof(bf16))
#define CONSUMER_THREADS_PER_GROUP 128
#define NUM_CONSUMER_GROUPS 2
#define CONSUMER_THREADS (NUM_CONSUMER_GROUPS * CONSUMER_THREADS_PER_GROUP)
#define NUM_THREADS (CONSUMER_THREADS + 1)  // 1 producer thread
#define BARRIER_SIZE_BYTES 8

__device__ __forceinline__ int32_t swizzle_128b(int32_t row, int32_t col) {
    return col ^ ((row & 7) << 3);
}

__global__ void __launch_bounds__(NUM_THREADS, 1) h100_matmul(
    __grid_constant__ const CUtensorMap a_map,
    __grid_constant__ const CUtensorMap b_map,
    __grid_constant__ const CUtensorMap c_map,
    int M, int N, int K
) {
    extern __shared__ char smem_buffer[];

    bf16* tile_buffer0 = reinterpret_cast<bf16*>(smem_buffer);
    bf16* tile_buffer1 = reinterpret_cast<bf16*>(smem_buffer + TILE_SZ_BYTES);
    bf16* output_tile = reinterpret_cast<bf16*>(smem_buffer + TILE_SZ_BYTES * 2);

    bf16* buf0_A = tile_buffer0;
    bf16* buf0_B = tile_buffer0 + (TILE_M * TILE_K);
    bf16* buf1_A = tile_buffer1;
    bf16* buf1_B = tile_buffer1 + (TILE_M * TILE_K);

    uint32_t tile_m = blockIdx.y * TILE_M;
    uint32_t tile_n = blockIdx.x * TILE_N;

    size_t barrier_start = TILE_SZ_BYTES * 2 + OUT_TILE_SZ_BYTES;
    uint64_t* prod_bar0 = reinterpret_cast<uint64_t*>(
        smem_buffer + barrier_start + 0 * BARRIER_SIZE_BYTES);
    uint64_t* cons_bar0 = reinterpret_cast<uint64_t*>(
        smem_buffer + barrier_start + 1 * BARRIER_SIZE_BYTES);
    uint64_t* prod_bar1 = reinterpret_cast<uint64_t*>(
        smem_buffer + barrier_start + 2 * BARRIER_SIZE_BYTES);
    uint64_t* cons_bar1 = reinterpret_cast<uint64_t*>(
        smem_buffer + barrier_start + 3 * BARRIER_SIZE_BYTES);

    bool is_producer = (threadIdx.x == CONSUMER_THREADS);

    if (threadIdx.x == 0) {
        init_barrier(prod_bar0, 1);
        init_barrier(cons_bar0, CONSUMER_THREADS);
        init_barrier(prod_bar1, 1);
        init_barrier(cons_bar1, CONSUMER_THREADS);
    }
    __syncthreads();

    uint32_t phase = 0;

    if (is_producer) {
        // Reduce register pressure for producer
        warpgroup_reg_dealloc<24>();

        for (uint32_t tile_pair = 0; tile_pair < (K / TILE_K) / 2; tile_pair++) {
            uint32_t tile0_k = 2 * tile_pair * TILE_K;
            uint32_t tile1_k = (2 * tile_pair + 1) * TILE_K;

            // Load buffer 0
            wait(cons_bar0, phase);
            expect_bytes_and_arrive(prod_bar0, TILE_SZ_BYTES);
            async_proxy_fence();

            cp_async_bulk_tensor_2d_global_to_shared(
                buf0_A, &a_map, tile0_k, tile_m, prod_bar0
            );
            cp_async_bulk_tensor_2d_global_to_shared(
                buf0_B, &b_map, tile0_k, tile_n, prod_bar0
            );

            // Load buffer 1
            wait(cons_bar1, phase);
            expect_bytes_and_arrive(prod_bar1, TILE_SZ_BYTES);
            async_proxy_fence();

            cp_async_bulk_tensor_2d_global_to_shared(
                buf1_A, &a_map, tile1_k, tile_m, prod_bar1
            );
            cp_async_bulk_tensor_2d_global_to_shared(
                buf1_B, &b_map, tile1_k, tile_n, prod_bar1
            );

            phase = 1 - phase;
        }
    } else {
        // Allocate registers for WGMMA accumulation
        warpgroup_reg_alloc<160>();

        uint32_t consumer_wg_id = threadIdx.x / CONSUMER_THREADS_PER_GROUP;
        uint32_t local_thread_id = threadIdx.x % CONSUMER_THREADS_PER_GROUP;
        uint32_t warp_id = local_thread_id / 32;
        uint32_t lane_id = threadIdx.x % 32;

        // Each warpgroup operates on its own M-slice
        bf16* wg_buf0_A = buf0_A + consumer_wg_id * MMA_TILE_M * TILE_K;
        bf16* wg_buf1_A = buf1_A + consumer_wg_id * MMA_TILE_M * TILE_K;

        float accum[16][8];
        memset(accum, 0, sizeof(accum));

        // Signal initial readiness
        arrive(cons_bar0, 1);
        arrive(cons_bar1, 1);
        async_proxy_fence();

        for (uint32_t tile_pair = 0; tile_pair < (K / TILE_K) / 2; tile_pair++) {
            // Process buffer 0
            wait(prod_bar0, phase);
            warpgroup_arrive();
            
            for (int i = 0; i < TILE_K / MMA_TILE_K; i++) {
                uint64_t desc_a = make_smem_desc<SWIZZLE_128B>(
                    wg_buf0_A + i * MMA_TILE_K, 8 * 128, 8 * 128
                );
                uint64_t desc_b = make_smem_desc<SWIZZLE_128B>(
                    buf0_B + i * MMA_TILE_K, 8 * 128, 8 * 128
                );
                wgmma_n256<1, 1, 1, 0, 0>(desc_a, desc_b, accum);
            }

            wgmma_commit();
            wgmma_wait<0>();
            arrive(cons_bar0, 1);
            async_proxy_fence();

            // Process buffer 1
            wait(prod_bar1, phase);
            warpgroup_arrive();
            
            for (int i = 0; i < TILE_K / MMA_TILE_K; i++) {
                uint64_t desc_a = make_smem_desc<SWIZZLE_128B>(
                    wg_buf1_A + i * MMA_TILE_K, 8 * 128, 8 * 128
                );
                uint64_t desc_b = make_smem_desc<SWIZZLE_128B>(
                    buf1_B + i * MMA_TILE_K, 8 * 128, 8 * 128
                );
                wgmma_n256<1, 1, 1, 0, 0>(desc_a, desc_b, accum);
            }

            wgmma_commit();
            wgmma_wait<0>();
            arrive(cons_bar1, 1);
            async_proxy_fence();

            phase = 1 - phase;
        }

        // Store to shared memory with swizzling
        uint32_t local_m = 16 * warp_id + (lane_id / 4);
        uint32_t local_n = 2 * (lane_id % 4);

        for (int i = 0; i < 32; i++) {
            int base_idx = i * 4;
            int r0 = (base_idx + 0) / 8, c0 = (base_idx + 0) % 8;
            int r1 = (base_idx + 1) / 8, c1 = (base_idx + 1) % 8;
            int r2 = (base_idx + 2) / 8, c2 = (base_idx + 2) % 8;
            int r3 = (base_idx + 3) / 8, c3 = (base_idx + 3) % 8;

            int out_row_0 = local_n + i * 8;
            int out_row_1 = local_n + i * 8 + 1;
            int out_col_0 = local_m;
            int out_col_1 = local_m + 8;

            // Write to warpgroup's slice of output tile
            bf16* wg_output = output_tile + consumer_wg_id * MMA_TILE_M * TILE_N;
            
            wg_output[out_row_0 * MMA_TILE_M + swizzle_128b(out_row_0, out_col_0)] = 
                __float2bfloat16(accum[r0][c0]);
            wg_output[out_row_1 * MMA_TILE_M + swizzle_128b(out_row_1, out_col_0)] = 
                __float2bfloat16(accum[r1][c1]);
            wg_output[out_row_0 * MMA_TILE_M + swizzle_128b(out_row_0, out_col_1)] = 
                __float2bfloat16(accum[r2][c2]);
            wg_output[out_row_1 * MMA_TILE_M + swizzle_128b(out_row_1, out_col_1)] = 
                __float2bfloat16(accum[r3][c3]);
        }

        __syncthreads();

        // Use TMA to write back (one thread per warpgroup)
        if (local_thread_id == 0) {
            bf16* wg_output = output_tile + consumer_wg_id * MMA_TILE_M * TILE_N;
            uint32_t wg_tile_m = tile_m + consumer_wg_id * MMA_TILE_M;
            
            cp_async_bulk_tensor_2d_shared_to_global(
                &c_map, wg_tile_m, tile_n, wg_output
            );
        }

        tma_commit_group();
    }
}

void launch_h100_matmul(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C) {
    CUtensorMap a_map, b_map, c_map;

    // A TMA map (K, M)
    const cuuint64_t a_globalDim[2] = {K, M};
    const cuuint64_t a_globalStrides[1] = {K * sizeof(bf16)};
    uint32_t a_boxDim[2] = {TILE_K, TILE_M};
    uint32_t a_elementStrides[2] = {1, 1};

    CUDA_CHECK(cuTensorMapEncodeTiled(
        &a_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, A, a_globalDim,
        a_globalStrides, a_boxDim, a_elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    // B TMA map (K, N)
    const cuuint64_t b_globalDim[2] = {K, N};
    const cuuint64_t b_globalStrides[1] = {K * sizeof(bf16)};
    uint32_t b_boxDim[2] = {TILE_K, TILE_N};
    uint32_t b_elementStrides[2] = {1, 1};

    CUDA_CHECK(cuTensorMapEncodeTiled(
        &b_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, B, b_globalDim,
        b_globalStrides, b_boxDim, b_elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    // C TMA map (M, N)
    const cuuint64_t c_globalDim[2] = {M, N};
    const cuuint64_t c_globalStrides[1] = {M * sizeof(bf16)};
    uint32_t c_boxDim[2] = {MMA_TILE_M, TILE_N};
    uint32_t c_elementStrides[2] = {1, 1};

    CUDA_CHECK(cuTensorMapEncodeTiled(
        &c_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, C, c_globalDim,
        c_globalStrides, c_boxDim, c_elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    dim3 num_blocks((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M, 1);
    size_t shared_mem_size = TILE_SZ_BYTES * 2 + OUT_TILE_SZ_BYTES + BARRIER_SIZE_BYTES * 4;
    
    CUDA_CHECK(cudaFuncSetAttribute(
        h100_matmul,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem_size));
    
    h100_matmul<<<num_blocks, NUM_THREADS, shared_mem_size>>>(
        a_map, b_map, c_map, M, N, K);
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

bool check_permutation_subset(bf16 *ref, bf16 *test, int N, int num_samples = 100, float tolerance = 0.1f) {
    std::cout << "\n=== Checking if ref values exist in test ===" << std::endl;
    
    int found = 0;
    for (int sample = 0; sample < num_samples; sample++) {
        int ref_idx = sample;
        float ref_val = __bfloat162float(ref[ref_idx]);
        
        // Search for this value anywhere in test
        bool found_match = false;
        for (int test_idx = 0; test_idx < N; test_idx++) {
            float test_val = __bfloat162float(test[test_idx]);
            if (std::abs(test_val - ref_val) <= tolerance) {
                found_match = true;
                std::cout << "  ref[" << ref_idx << "]=" << ref_val 
                          << " found at test[" << test_idx << "]" << std::endl;
                break;
            }
        }
        if (found_match) found++;
        else {
            std::cout << "  ref[" << ref_idx << "]=" << ref_val 
                      << " NOT FOUND in test ✗" << std::endl;
        }
    }
    
    std::cout << "Found " << found << "/" << num_samples << " ref values in test" << std::endl;
    return found == num_samples;
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

    //bool permutation_subset = check_permutation_subset(hCublas, hOurs, M * N, 100, 0.01f);
    //printf("%s permutation subset output!\n\n\n", permutation_subset ? "Correct" : "Incorrect");

    long flops = 2LL * M * N * K;
    BENCHPRESS(runCublasRef, flops, M, N, K, dA, dB, dCublas);

    BENCHPRESS(launch_h100_matmul, flops, M, N, K, dA, dB, dC);

    free(hCublas);
    free(hOurs);

    return 0;
}
