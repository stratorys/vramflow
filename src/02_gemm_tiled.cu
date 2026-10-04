#include "common.cuh"

#include <cublas_v2.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <limits>
#include <vector>

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    const cublasStatus_t status = (call);                                      \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      std::fprintf(stderr, "%s:%d: %s: cuBLAS status %d\n", __FILE__,          \
                   __LINE__, #call, int(status));                              \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// A: [M,ceil(K/4)], BT: [N,ceil(K/4)], with signed four-byte lanes.
// A block produces 64x64 outputs, consuming 32 scalar K values per tile.
template <int SharedStride>
__global__ void gemm_tiled(const int *a, const int *bt, int32_t *c, int m,
                           int k4, int n) {
  static_assert(SharedStride == 65 || SharedStride == 68, "unsupported stride");
  // Stride 65 is the original baseline. Stride 68 maps the 32 stores of each
  // warp to distinct banks for local=index/8 and lane=index%8.
  __shared__ int shared_a[8][SharedStride];
  __shared__ int shared_b[8][SharedStride];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * 16 + tx;
  const int row_base = blockIdx.y * 64;
  const int col_base = blockIdx.x * 64;
  int acc[4][4] = {};

  for (int base = 0; base < k4; base += 8) {
#pragma unroll
    for (int load = 0; load < 2; ++load) {
      const int index = tid + load * 256;
      const int lane = index % 8;
      const int local = index / 8;
      const int group = base + lane;
      const int row = row_base + local;
      const int col = col_base + local;
      shared_a[lane][local] = row < m && group < k4 ? a[row * k4 + group] : 0;
      shared_b[lane][local] = col < n && group < k4 ? bt[col * k4 + group] : 0;
    }
    __syncthreads();

#pragma unroll
    for (int p = 0; p < 8; ++p) {
      int reg_a[4], reg_b[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        reg_a[i] = shared_a[p][ty + i * 16];
        reg_b[i] = shared_b[p][tx + i * 16];
      }
#pragma unroll
      for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          acc[i][j] = __dp4a(reg_a[i], reg_b[j], acc[i][j]);
        }
      }
    }
    // Even out-of-bounds output threads must reach both barriers.
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int row = row_base + ty + i * 16;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int col = col_base + tx + j * 16;
      if (row < m && col < n) {
        c[row * n + col] = acc[i][j];
      }
    }
  }
}

enum class Input { Random, Zero, Extremes };

static void fill_inputs(std::vector<int8_t> &a, std::vector<int8_t> &b,
                        Input input) {
  uint32_t state = 0x12345678u;
  for (auto *values : {&a, &b}) {
    for (size_t i = 0; i < values->size(); ++i) {
      if (input == Input::Zero) {
        (*values)[i] = 0;
      } else if (input == Input::Extremes) {
        constexpr int8_t pattern[] = {-128, 127, -1, 0, 1};
        (*values)[i] = pattern[i % 5];
      } else {
        state = state * 1664525u + 1013904223u;
        (*values)[i] = static_cast<int8_t>(int(state >> 28) - 8);
      }
    }
  }
}

static void pack_inputs(const std::vector<int8_t> &a,
                        const std::vector<int8_t> &b,
                        std::vector<int> &packed_a, std::vector<int> &packed_bt,
                        int m, int k, int n) {
  const int k4 = (k + 3) / 4;
  for (int row = 0; row < m; ++row) {
    for (int group = 0; group < k4; ++group) {
      int8_t lanes[4] = {};
      for (int lane = 0; lane < 4; ++lane) {
        const int p = group * 4 + lane;
        if (p < k)
          lanes[lane] = a[row * k + p];
      }
      packed_a[row * k4 + group] =
          pack4(lanes[0], lanes[1], lanes[2], lanes[3]);
    }
  }
  for (int col = 0; col < n; ++col) {
    for (int group = 0; group < k4; ++group) {
      int8_t lanes[4] = {};
      for (int lane = 0; lane < 4; ++lane) {
        const int p = group * 4 + lane;
        if (p < k)
          lanes[lane] = b[p * n + col];
      }
      packed_bt[col * k4 + group] =
          pack4(lanes[0], lanes[1], lanes[2], lanes[3]);
    }
  }
}

static bool verify_element(const std::vector<int8_t> &a,
                           const std::vector<int8_t> &b,
                           const std::vector<int32_t> &c, int row, int col,
                           int k, int n) {
  int64_t expected = 0;
  for (int p = 0; p < k; ++p) {
    expected += int64_t(a[row * k + p]) * int64_t(b[p * n + col]);
  }
  if (expected < std::numeric_limits<int32_t>::min() ||
      expected > std::numeric_limits<int32_t>::max() ||
      c[row * n + col] != expected) {
    std::fprintf(stderr, "CPU mismatch at C[%d,%d]: tiled=%d CPU=%lld\n", row,
                 col, int(c[row * n + col]), static_cast<long long>(expected));
    return false;
  }
  return true;
}

static bool verify_cpu(const std::vector<int8_t> &a,
                       const std::vector<int8_t> &b,
                       const std::vector<int32_t> &c, int m, int k, int n,
                       bool sampled) {
  if (!sampled) {
    for (int row = 0; row < m; ++row) {
      for (int col = 0; col < n; ++col) {
        if (!verify_element(a, b, c, row, col, k, n))
          return false;
      }
    }
  } else {
    // Include boundaries, then reproducibly sample the remaining positions.
    uint32_t state = 0x9e3779b9u;
    for (int i = 0; i < 64; ++i) {
      state = state * 1664525u + 1013904223u;
      int row = int((state >> 8) % uint32_t(m));
      state = state * 1664525u + 1013904223u;
      int col = int((state >> 8) % uint32_t(n));
      if (i < 4) {
        row = (i & 1) ? m - 1 : 0;
        col = (i & 2) ? n - 1 : 0;
      }
      if (!verify_element(a, b, c, row, col, k, n))
        return false;
    }
  }
  return true;
}

template <typename Launch> static double median_ms(Launch launch) {
  for (int i = 0; i < 5; ++i)
    launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  std::array<float, 20> times{};
  for (float &ms : times) {
    CUDA_CHECK(cudaEventRecord(start));
    launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  std::sort(times.begin(), times.end());
  const double ms = (double(times[9]) + times[10]) / 2.0;
  if (ms <= 0.0) {
    std::fprintf(stderr, "Invalid CUDA event duration: %.6f ms\n", ms);
    std::exit(EXIT_FAILURE);
  }
  return ms;
}

static bool run_case(cublasHandle_t handle, int m, int k, int n, Input input,
                     bool benchmark) {
  const int k4 = (k + 3) / 4;
  std::vector<int8_t> a(size_t(m) * k), b(size_t(k) * n);
  std::vector<int> packed_a(size_t(m) * k4), packed_bt(size_t(n) * k4);
  std::vector<int32_t> c(size_t(m) * n);
  fill_inputs(a, b, input);
  const auto pack_start = std::chrono::steady_clock::now();
  pack_inputs(a, b, packed_a, packed_bt, m, k, n);
  const double pack_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - pack_start)
                             .count();

  int *d_packed_a = nullptr, *d_packed_bt = nullptr;
  int32_t *d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_packed_a, packed_a.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_packed_bt, packed_bt.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_c, c.size() * sizeof(int32_t)));
  CUDA_CHECK(cudaMemcpy(d_packed_a, packed_a.data(),
                        packed_a.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_packed_bt, packed_bt.data(),
                        packed_bt.size() * sizeof(int),
                        cudaMemcpyHostToDevice));
  const dim3 block(16, 16), grid((n + 63) / 64, (m + 63) / 64);
  const auto launch_tiled = [&](int stride) {
    if (stride == 65) {
      gemm_tiled<65><<<grid, block>>>(d_packed_a, d_packed_bt, d_c, m, k4, n);
    } else {
      gemm_tiled<68><<<grid, block>>>(d_packed_a, d_packed_bt, d_c, m, k4, n);
    }
    CUDA_CHECK(cudaGetLastError());
  };
  const char *label = input == Input::Random ? "random"
                      : input == Input::Zero ? "zeros"
                                             : "extremes";
  std::printf("\nM=%d K=%d N=%d input=%s\n", m, k, n, label);
  const auto validate_tiled = [&](int stride) {
    // Poison the reused output so omitted writes cannot inherit a prior result.
    CUDA_CHECK(cudaMemset(d_c, 0x80, c.size() * sizeof(int32_t)));
    launch_tiled(stride);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(c.data(), d_c, c.size() * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));
    const bool valid = verify_cpu(a, b, c, m, k, n, benchmark);
    std::printf("stride=%d CPU reference (%s): %s\n", stride,
                benchmark ? "64 positions" : "all outputs",
                valid ? "PASS" : "FAIL");
    return valid;
  };
  bool correct = validate_tiled(65);

  // The original row-major matrices satisfy cuBLAS alignment constraints only
  // for these shapes. Odd-sized correctness cases use the CPU reference alone.
  const bool compare_blas = m % 4 == 0 && k % 4 == 0 && n % 4 == 0;
  int8_t *d_a = nullptr, *d_b = nullptr;
  int32_t *d_blas_c = nullptr;
  if (correct && compare_blas) {
    CUDA_CHECK(cudaMalloc(&d_a, a.size() * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_b, b.size() * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_blas_c, c.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), a.size() * sizeof(int8_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), b.size() * sizeof(int8_t),
                          cudaMemcpyHostToDevice));
    const int32_t alpha = 1, beta = 0;
    const auto launch_blas = [&]() {
      // Column-major C^T[N,M] = B^T[N,K] * A^T[K,M].
      CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                                &alpha, d_b, CUDA_R_8I, n, d_a, CUDA_R_8I, k,
                                &beta, d_blas_c, CUDA_R_32I, n,
                                CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
    };
    launch_blas();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<int32_t> blas_c(c.size());
    CUDA_CHECK(cudaMemcpy(blas_c.data(), d_blas_c,
                          blas_c.size() * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));
    const auto compare_outputs = [&](int stride) {
      bool valid = true;
      for (size_t i = 0; i < c.size(); ++i) {
        if (c[i] != blas_c[i]) {
          std::fprintf(
              stderr,
              "stride=%d cuBLAS mismatch at C[%zu,%zu]: tiled=%d cuBLAS=%d\n",
              stride, i / n, i % n, int(c[i]), int(blas_c[i]));
          valid = false;
          break;
        }
      }
      std::printf("stride=%d cuBLAS reference (all outputs): %s\n", stride,
                  valid ? "PASS" : "FAIL");
      return valid;
    };
    correct = compare_outputs(65) && validate_tiled(68) && compare_outputs(68);
    if (correct && benchmark) {
      const double baseline_ms = median_ms([&]() { launch_tiled(65); });
      const double padded_ms = median_ms([&]() { launch_tiled(68); });
      const double blas_ms = median_ms(launch_blas);
      const double operations = 2.0 * m * n * k;
      std::printf("CPU packing (excluded): %.3f ms\n", pack_ms);
      std::printf("tiled stride=65: %.4f ms  %.4f TOPS\n", baseline_ms,
                  operations / (baseline_ms * 1e9));
      std::printf("tiled stride=68: %.4f ms  %.4f TOPS\n", padded_ms,
                  operations / (padded_ms * 1e9));
      std::printf("cuBLAS: %.4f ms  %.4f TOPS\n", blas_ms,
                  operations / (blas_ms * 1e9));
      std::printf("stride=68 / stride=65 throughput: %.3f (speedup)\n",
                  baseline_ms / padded_ms);
      std::printf("stride=65 / cuBLAS throughput: %.3f (%.1f%%)\n",
                  blas_ms / baseline_ms, 100.0 * blas_ms / baseline_ms);
      std::printf("stride=68 / cuBLAS throughput: %.3f (%.1f%%)\n",
                  blas_ms / padded_ms, 100.0 * blas_ms / padded_ms);
    }
  } else if (correct) {
    correct = validate_tiled(68);
    std::printf("cuBLAS reference: skipped (unaligned shape)\n");
  }

  if (d_blas_c)
    CUDA_CHECK(cudaFree(d_blas_c));
  if (d_b)
    CUDA_CHECK(cudaFree(d_b));
  if (d_a)
    CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_c));
  CUDA_CHECK(cudaFree(d_packed_bt));
  CUDA_CHECK(cudaFree(d_packed_a));
  return correct;
}

int main(int argc, char **argv) {
  bool check = true, bench = true;
  if (argc == 2 && std::strcmp(argv[1], "--check") == 0) {
    bench = false;
  } else if (argc == 2 && std::strcmp(argv[1], "--bench") == 0) {
    check = false;
  } else if (argc != 1) {
    std::fprintf(stderr, "Usage: %s [--check | --bench]\n", argv[0]);
    return EXIT_FAILURE;
  }

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU: %s\nCompute capability: %d.%d\n", prop.name, prop.major,
              prop.minor);
  if (prop.major < 6 || (prop.major == 6 && prop.minor < 1)) {
    std::fprintf(stderr, "DP4A requires compute capability 6.1 or newer.\n");
    return EXIT_FAILURE;
  }
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  CUBLAS_CHECK(cublasSetStream(handle, nullptr));
  int version = 0;
  CUBLAS_CHECK(cublasGetVersion(handle, &version));
  std::printf(
      "cuBLAS version: %d\nTile: 64x64x32, 16x16 threads, 4x4 outputs/thread\n",
      version);
  std::printf("Shared strides: 65 (baseline), 68 (store-bank padding)\n");
  bool correct = true;
  if (check) {
    std::printf("\nCorrectness tests\n");
    correct = run_case(handle, 17, 19, 23, Input::Random, false) &&
              run_case(handle, 65, 35, 67, Input::Random, false) &&
              run_case(handle, 65, 35, 67, Input::Zero, false) &&
              run_case(handle, 65, 35, 67, Input::Extremes, false) &&
              run_case(handle, 128, 65, 129, Input::Random, false);
    for (int k = 1; correct && k <= 7; ++k) {
      correct = run_case(handle, 3, k, 5, Input::Extremes, false);
    }
    for (int size : {128, 256, 512}) {
      if (!correct)
        break;
      correct = run_case(handle, size, size, size, Input::Random, false);
    }
  }
  if (correct && bench) {
    std::printf(
        "\nBenchmarks: 5 warmups, 20 timings per backend; GPU compute only\n");
    constexpr int shapes[][3] = {{1024, 1024, 1024},  {2048, 2048, 2048},
                                 {4096, 4096, 4096},  {8192, 4096, 4096},
                                 {8192, 4096, 16384}, {8192, 16384, 4096}};
    for (const auto &shape : shapes) {
      correct =
          run_case(handle, shape[0], shape[1], shape[2], Input::Random, true);
      if (!correct)
        break;
    }
  }
  CUBLAS_CHECK(cublasDestroy(handle));
  std::printf("\n%s: %s\n", correct ? "PASS" : "FAIL",
              correct ? "all requested correctness checks passed; performance "
                        "gate requires review"
                      : "numerical validation failed");
  return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
