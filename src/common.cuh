#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    const cudaError_t error = (call);                                          \
    if (error != cudaSuccess) {                                                \
      std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call,       \
                   cudaGetErrorString(error));                                 \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// Keep the result signed so __dp4a uses signed INT8 lanes.
__host__ __device__ inline int pack4(int8_t a, int8_t b, int8_t c, int8_t d) {
  const uint32_t bits = uint32_t(uint8_t(a)) | (uint32_t(uint8_t(b)) << 8) |
                        (uint32_t(uint8_t(c)) << 16) |
                        (uint32_t(uint8_t(d)) << 24);
  return static_cast<int>(bits);
}
