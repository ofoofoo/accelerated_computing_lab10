#pragma once

#include <cuda.h>
#include <iostream>

////////////////////////////////////////////////////////////////////////////////
// HELPER FUNCTION TO CHECK FOR ERRORS
////////////////////////////////////////////////////////////////////////////////
void cuda_check(CUresult code, const char *file, int line) {
  if (code != CUDA_SUCCESS) {
    char const *str;
    cuGetErrorString(code, &str);
    std::cerr << "CUDA error at " << file << ":" << line << ": " << str
              << std::endl;
    exit(1);
  }
}

void cuda_check(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    std::cerr << "CUDA error at " << file << ":" << line << ": "
              << cudaGetErrorString(code) << std::endl;
    exit(1);
  }
}

// Macro for convenient CUDA error checking
#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cuda_check((x), __FILE__, __LINE__);                                       \
  } while (0)

////////////////////////////////////////////////////////////////////////////////
// ASYNC PROXY FENCE
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Inserts a fence to ensure that memory operations in the generic proxy
 * have been made visible to the async proxy.
 *
 * @return This function does not return a value.
 */
__device__ static __forceinline__ void async_proxy_fence() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

////////////////////////////////////////////////////////////////////////////////
// MBARRIER FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Initializes a barrier in shared memory with arrival count.
 *
 * @param bar Pointer to the mbarrier in shared memory to initialize.
 * @param arrival_count The number of arrivals expected before the barrier
 * completes.
 * @note bar must be in shared memory, and 8-byte aligned.
 * @return This function does not return a value.
 */
__device__ static __forceinline__ void init_barrier(uint64_t *bar,
                                                    int arrival_count) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar_ptr),
               "r"(arrival_count)
               : "memory");
}

/**
 * @brief Arrive on an initialized mbarrier in shared memory and increment its
 * arrival count by count.
 *
 * @param bar Pointer to the mbarrier in shared memory.
 * @param count The number to add to the arrival count.
 * @return This function does not return a value.
 */
__device__ static __forceinline__ void arrive(uint64_t *bar, uint32_t count) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0],  %1;\n"
               :
               : "r"(mbar_ptr), "r"(count)
               : "memory");
}

/**
 * @brief Polls for current mbarrier phase. This function is non-blocking.
 *
 * @param bar Pointer to the mbarrier in shared memory.
 * @param phaseParity The expected phase parity to wait for.
 * @return Returns 1 if phase is completed, 0 otherwise.
 */
__device__ static __forceinline__ int try_wait(uint64_t *bar, int phaseParity) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  int result;
  asm volatile("{\n"
               ".reg .pred P1;\n"
               "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\n"
               "selp.u32 %0,1,0,P1;"
               "}\n"
               : "=r"(result)
               : "r"(mbar_ptr), "r"(phaseParity));
  return result;
}

/**
 * @brief Tests if phase is complete. May block based on hardware based timeout.
 *
 * @param bar Pointer to the mbarrier in shared memory.
 * @param phaseParity The expected phase parity to wait for.
 * @note There is an optional parameter that can set the timeout, but we do not
 * expose that.
 * @return Returns 1 if phase is completed, 0 otherwise.
 */
__device__ static __forceinline__ int test_wait(uint64_t *bar,
                                                int phaseParity) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  int result;
  asm volatile("{\n"
               ".reg .pred P1;\n"
               "mbarrier.test_wait.parity.shared::cta.b64 P1, [%1], %2;\n"
               "selp.u32 %0,1,0,P1;"
               "}\n"
               : "=r"(result)
               : "r"(mbar_ptr), "r"(phaseParity));
  return result;
}

/**
 * @brief Blocking wait until phase completes.
 *
 * @param bar Pointer to the mbarrier in shared memory.
 * @param phaseParity The expected phase parity to wait for.
 * @return This function does not return a value.
 */
__device__ static __forceinline__ void wait(uint64_t *bar, int phaseParity) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("{\n"
               ".reg .pred                P1;\n"
               "LAB_WAIT:\n"
               "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
               "@P1                       bra.uni DONE;\n"
               "bra.uni                   LAB_WAIT;\n"
               "DONE:\n"
               "}\n" ::"r"(mbar_ptr),
               "r"(phaseParity));
}

/**
 * @brief Add expected bytes to the barrier.
 *
 * @param bar Pointer to the mbarrier in shared memory.
 * @param bytes The number of bytes expected to arrive at the barrier.
 * @return This function does not return a value.
 */
__device__ static __forceinline__ void expect_bytes(uint64_t *bar,
                                                    uint32_t bytes) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;\n"
      :
      : "r"(bar_ptr), "r"(bytes)
      : "memory");
}

/**
 * @brief First add expected bytes to the barrier, and then arrive.
 *
 * @param bar Pointer to the mbarrier in shared memory.
 * @param bytes The number of bytes expected to arrive at the barrier.
 * @return This function does not return a value.
 */
__device__ static __forceinline__ void expect_bytes_and_arrive(uint64_t *bar,
                                                               uint32_t bytes) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm("mbarrier.arrive.expect_tx.release.cta.shared.b64 _, [%0], %1;\n "
      :
      : "r"(bar_ptr), "r"(bytes)
      : "memory");
}

////////////////////////////////////////////////////////////////////////////////
// TMA GROUP OPERATIONS
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Commit TMA bulk group.
 *
 * @note Only shared -> global uses .bulk_group completion.
 * @return This function does not return a value.
 */
__device__ static __forceinline__ void tma_commit_group() {
  asm volatile("cp.async.bulk.commit_group;");
}

/**
 * @brief Wait until at most N TMA groups remain pending.
 *
 * @tparam N The maximum number of TMA groups that can remain pending.
 * @return This function does not return a value.
 */
template <int N>
__device__ static __forceinline__ void tma_wait_until_pending() {
  asm volatile("cp.async.bulk.wait_group %0;" : : "n"(N) : "memory");
}

////////////////////////////////////////////////////////////////////////////////
// GLOBAL -> SHARED
////////////////////////////////////////////////////////////////////////////////

// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor

/**
 * @brief Asynchronously copy a 1D tensor tile from global to shared memory.
 *
 * @param smem_dest Destination address in shared memory.
 * @param tensor_map Tensor map descriptor for the source tensor.
 * @param c0 Coordinate in the first dimension.
 * @param bar Pointer to mbarrier for completion tracking.
 * @return This function does not return a value.
 */
__device__ static __forceinline__ void cp_async_bulk_tensor_1d_global_to_shared(
    void *smem_dest, const CUtensorMap *tensor_map, int c0, uint64_t *bar) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_"
      "tx::bytes "
      "[%0], [%1, {%2}], [%3];\n"
      :
      : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem_dest))),
        "l"(tensor_map), "r"(c0), "r"(mbar_ptr)
      : "memory");
}

/**
 * @brief Asynchronously copy a 2D tensor tile from global to shared memory.
 *
 * @param smem_dest Destination address in shared memory.
 * @param tensor_map Tensor map descriptor for the source tensor.
 * @param c0 Coordinate in the first dimension.
 * @param c1 Coordinate in the second dimension.
 * @param bar Pointer to mbarrier for completion tracking.
 * @return This function does not return a value.
 */
__device__ static __forceinline__ void
cp_async_bulk_tensor_2d_global_to_shared(void *smem_dest,
                                         const CUtensorMap *tensor_map, int c0,
                                         int c1, uint64_t *bar) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_"
      "tx::bytes "
      "[%0], [%1, {%2, %3}], [%4];\n"
      :
      : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem_dest))),
        "l"(tensor_map), "r"(c0), "r"(c1), "r"(mbar_ptr)
      : "memory");
}

/**
 * @brief Asynchronously copy a 3D tensor tile from global to shared memory.
 *
 * @param smem_dest Destination address in shared memory.
 * @param tensor_map Tensor map descriptor for the source tensor.
 * @param c0 Coordinate in the first dimension.
 * @param c1 Coordinate in the second dimension.
 * @param c2 Coordinate in the third dimension.
 * @param bar Pointer to mbarrier for completion tracking.
 * @return This function does not return a value.
 */
__device__ static __forceinline__ void
cp_async_bulk_tensor_3d_global_to_shared(void *smem_dest,
                                         const CUtensorMap *tensor_map, int c0,
                                         int c1, int c2, uint64_t *bar) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_"
      "tx::bytes "
      "[%0], [%1, {%2, %3, %4}], [%5];\n"
      :
      : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem_dest))),
        "l"(tensor_map), "r"(c0), "r"(c1), "r"(c2), "r"(mbar_ptr)
      : "memory");
}

/**
 * @brief Asynchronously copy a 4D tensor tile from global to shared memory.
 *
 * @param smem_dest Destination address in shared memory.
 * @param tensor_map Tensor map descriptor for the source tensor.
 * @param c0 Coordinate in the first dimension.
 * @param c1 Coordinate in the second dimension.
 * @param c2 Coordinate in the third dimension.
 * @param c3 Coordinate in the fourth dimension.
 * @param bar Pointer to mbarrier for completion tracking.
 * @return This function does not return a value.
 */
__device__ static __forceinline__ void cp_async_bulk_tensor_4d_global_to_shared(
    void *smem_dest, const CUtensorMap *tensor_map, int c0, int c1, int c2,
    int c3, uint64_t *bar) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_"
      "tx::bytes "
      "[%0], [%1, {%2, %3, %4, %5}], [%6];\n"
      :
      : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem_dest))),
        "l"(tensor_map), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(mbar_ptr)
      : "memory");
}

/**
 * @brief Asynchronously copy a 5D tensor tile from global to shared memory.
 *
 * @param smem_dest Destination address in shared memory.
 * @param tensor_map Tensor map descriptor for the source tensor.
 * @param c0 Coordinate in the first dimension.
 * @param c1 Coordinate in the second dimension.
 * @param c2 Coordinate in the third dimension.
 * @param c3 Coordinate in the fourth dimension.
 * @param c4 Coordinate in the fifth dimension.
 * @param bar Pointer to mbarrier for completion tracking.
 * @return This function does not return a value.
 */
__device__ static __forceinline__ void cp_async_bulk_tensor_5d_global_to_shared(
    void *smem_dest, const CUtensorMap *tensor_map, int c0, int c1, int c2,
    int c3, int c4, uint64_t *bar) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_"
      "tx::bytes "
      "[%0], [%1, {%2, %3, %4, %5, %6}], [%7];\n"
      :
      : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem_dest))),
        "l"(tensor_map), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4),
        "r"(mbar_ptr)
      : "memory");
}

////////////////////////////////////////////////////////////////////////////////
// SHARED -> GLOBAL
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Asynchronously copy a 1D tensor tile from shared to global memory.
 *
 * @param tensor_map Tensor map descriptor for the destination tensor.
 * @param c0 Coordinate in the first dimension.
 * @param src Source address in shared memory.
 * @return This function does not return a value.
 */
__device__ static __forceinline__ void
cp_async_bulk_tensor_1d_shared_to_global(const CUtensorMap *tensor_map, int c0,
                                         const void *src) {
  asm volatile("cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group "
               "[%0, {%1}], [%2];\n"
               :
               : "l"(tensor_map), "r"(c0),
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(src)))
               : "memory");
}

/**
 * @brief Asynchronously copy a 2D tensor tile from shared to global memory.
 *
 * @param tensor_map Tensor map descriptor for the destination tensor.
 * @param c0 Coordinate in the first dimension.
 * @param c1 Coordinate in the second dimension.
 * @param src Source address in shared memory.
 * @return This function does not return a value.
 */
__device__ static __forceinline__ void
cp_async_bulk_tensor_2d_shared_to_global(const CUtensorMap *tensor_map, int c0,
                                         int c1, const void *src) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group "
               "[%0, {%1, %2}], [%3];\n"
               :
               : "l"(tensor_map), "r"(c0), "r"(c1),
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(src)))
               : "memory");
}

/**
 * @brief Asynchronously copy a 3D tensor tile from shared to global memory.
 *
 * @param tensor_map Tensor map descriptor for the destination tensor.
 * @param c0 Coordinate in the first dimension.
 * @param c1 Coordinate in the second dimension.
 * @param c2 Coordinate in the third dimension.
 * @param src Source address in shared memory.
 * @return This function does not return a value.
 */
__device__ static __forceinline__ void
cp_async_bulk_tensor_3d_shared_to_global(const CUtensorMap *tensor_map, int c0,
                                         int c1, int c2, const void *src) {
  asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group "
               "[%0, {%1, %2, %3}], [%4];\n"
               :
               : "l"(tensor_map), "r"(c0), "r"(c1), "r"(c2),
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(src)))
               : "memory");
}

/**
 * @brief Asynchronously copy a 4D tensor tile from shared to global memory.
 *
 * @param tensor_map Tensor map descriptor for the destination tensor.
 * @param c0 Coordinate in the first dimension.
 * @param c1 Coordinate in the second dimension.
 * @param c2 Coordinate in the third dimension.
 * @param c3 Coordinate in the fourth dimension.
 * @param src Source address in shared memory.
 * @return This function does not return a value.
 */
__device__ static __forceinline__ void
cp_async_bulk_tensor_4d_shared_to_global(const CUtensorMap *tensor_map, int c0,
                                         int c1, int c2, int c3,
                                         const void *src) {
  asm volatile("cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group "
               "[%0, {%1, %2, %3, %4}], [%5];\n"
               :
               : "l"(tensor_map), "r"(c0), "r"(c1), "r"(c2), "r"(c3),
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(src)))
               : "memory");
}

/**
 * @brief Asynchronously copy a 5D tensor tile from shared to global memory.
 *
 * @param tensor_map Tensor map descriptor for the destination tensor.
 * @param c0 Coordinate in the first dimension.
 * @param c1 Coordinate in the second dimension.
 * @param c2 Coordinate in the third dimension.
 * @param c3 Coordinate in the fourth dimension.
 * @param c4 Coordinate in the fifth dimension.
 * @param src Source address in shared memory.
 * @return This function does not return a value.
 */
__device__ static __forceinline__ void
cp_async_bulk_tensor_5d_shared_to_global(const CUtensorMap *tensor_map, int c0,
                                         int c1, int c2, int c3, int c4,
                                         const void *src) {
  asm volatile("cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group "
               "[%0, {%1, %2, %3, %4, %5}], [%6];\n"
               :
               : "l"(tensor_map), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4),
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(src)))
               : "memory");
}