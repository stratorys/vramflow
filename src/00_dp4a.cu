#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t err = (x);                                                     \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));            \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static int pack4(int8_t a, int8_t b, int8_t c, int8_t d) {
  return ((uint32_t)(uint8_t)a) | ((uint32_t)(uint8_t)b << 8) |
         ((uint32_t)(uint8_t)c << 16) | ((uint32_t)(uint8_t)d << 24);
}

__global__ void test_dp4a(int a, int b, int *out) {
  int acc = 0;
  acc = __dp4a(a, b, acc);
  *out = acc;
}

__global__ void bench_dp4a(int a, int b, int iterations, int *out) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int acc0 = tid;
  int acc1 = tid + 1;
  int acc2 = tid + 2;
  int acc3 = tid + 3;

#pragma unroll 1
  for (int i = 0; i < iterations; ++i) {
    acc0 = __dp4a(a, b, acc0);
    acc1 = __dp4a(a, b, acc1);
    acc2 = __dp4a(a, b, acc2);
    acc3 = __dp4a(a, b, acc3);
  }

  out[tid] = acc0 + acc1 + acc2 + acc3;
}

int main() {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  printf("GPU: %s\n", prop.name);
  printf("Compute capability: %d.%d\n\n", prop.major, prop.minor);

  int a = pack4(1, -2, 3, -4);
  int b = pack4(5, 6, -7, 8);

  // 1*5 + -2*6 + 3*-7 + -4*8 = -60

  int *d_result;
  CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));

  test_dp4a<<<1, 1>>>(a, b, d_result);
  CUDA_CHECK(cudaDeviceSynchronize());

  int result;
  CUDA_CHECK(
      cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost));

  printf("Correctness\n");
  printf("GPU      : %d\n", result);
  printf("Expected : -60\n");

  if (result != -60) {
    printf("FAIL\n");
    return 1;
  }

  printf("PASS\n\n");

  constexpr int blocks = 256;
  constexpr int threads = 256;
  constexpr int total_threads = blocks * threads;
  constexpr int iterations = 8192;

  int *d_out;
  CUDA_CHECK(cudaMalloc(&d_out, total_threads * sizeof(int)));

  // warmup
  bench_dp4a<<<blocks, threads>>>(a, b, 256, d_out);

  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));

  bench_dp4a<<<blocks, threads>>>(a, b, iterations, d_out);

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  double dp4a_count = (double)total_threads * iterations * 4.0;

  // chaque DP4A = 4 MAC INT8
  double mac_count = dp4a_count * 4.0;

  double seconds = ms / 1000.0;

  double ginst = dp4a_count / seconds / 1e9;

  double tmac = mac_count / seconds / 1e12;

  // convention 1 multiply + 1 add = 2 ops
  double tops = mac_count * 2.0 / seconds / 1e12;

  printf("Benchmark\n");
  printf("Time      : %.3f ms\n", ms);
  printf("DP4A      : %.2f Ginst/s\n", ginst);
  printf("INT8 MAC  : %.2f TMAC/s\n", tmac);
  printf("INT8 TOPS : %.2f TOPS\n", tops);

  CUDA_CHECK(cudaFree(d_result));
  CUDA_CHECK(cudaFree(d_out));

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
