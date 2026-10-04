#include "common.cuh"

#include <algorithm>
#include <array>
#include <limits>
#include <vector>

// Row-major A[M,K] * B[K,N] = C[M,N]. One thread owns one output.
__global__ void gemm_naive(const int8_t *a, const int8_t *b, int32_t *c, int m,
                           int k, int n) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || col >= n) {
    return;
  }

  int acc = 0;
  int p = 0;
  for (; p + 3 < k; p += 4) {
    const int packed_a = pack4(a[row * k + p], a[row * k + p + 1],
                               a[row * k + p + 2], a[row * k + p + 3]);
    const int packed_b = pack4(b[p * n + col], b[(p + 1) * n + col],
                               b[(p + 2) * n + col], b[(p + 3) * n + col]);
    acc = __dp4a(packed_a, packed_b, acc);
  }
  for (; p < k; ++p) {
    acc += int(a[row * k + p]) * int(b[p * n + col]);
  }
  c[row * n + col] = acc;
}

enum class Input { Random, Zero, Extremes };

// A fixed unsigned generator keeps inputs reproducible across platforms.
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

static bool verify(const std::vector<int8_t> &a, const std::vector<int8_t> &b,
                   const std::vector<int32_t> &c, int m, int k, int n) {
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      int64_t expected = 0;
      for (int p = 0; p < k; ++p) {
        expected += int64_t(a[row * k + p]) * int64_t(b[p * n + col]);
      }
      if (expected < std::numeric_limits<int32_t>::min() ||
          expected > std::numeric_limits<int32_t>::max() ||
          c[row * n + col] != expected) {
        std::fprintf(stderr, "Mismatch at C[%d,%d]: GPU=%d CPU=%lld\n", row,
                     col, int(c[row * n + col]),
                     static_cast<long long>(expected));
        return false;
      }
    }
  }
  return true;
}

static bool run_case(int m, int k, int n, Input input, bool benchmark) {
  std::vector<int8_t> a(size_t(m) * k), b(size_t(k) * n);
  std::vector<int32_t> c(size_t(m) * n);
  fill_inputs(a, b, input);

  int8_t *d_a = nullptr, *d_b = nullptr;
  int32_t *d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, a.size() * sizeof(int8_t)));
  CUDA_CHECK(cudaMalloc(&d_b, b.size() * sizeof(int8_t)));
  CUDA_CHECK(cudaMalloc(&d_c, c.size() * sizeof(int32_t)));
  CUDA_CHECK(cudaMemcpy(d_a, a.data(), a.size() * sizeof(int8_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b.data(), b.size() * sizeof(int8_t),
                        cudaMemcpyHostToDevice));

  const dim3 block(16, 16);
  const dim3 grid((n + 15) / 16, (m + 15) / 16);
  const auto launch = [&]() {
    gemm_naive<<<grid, block>>>(d_a, d_b, d_c, m, k, n);
    CUDA_CHECK(cudaGetLastError());
  };
  launch();
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(c.data(), d_c, c.size() * sizeof(int32_t),
                        cudaMemcpyDeviceToHost));
  const bool correct = verify(a, b, c, m, k, n);
  const char *label = input == Input::Random ? "random"
                      : input == Input::Zero ? "zeros"
                                             : "extremes";
  std::printf("M=%d K=%d N=%d input=%s correct: %s\n", m, k, n, label,
              correct ? "YES" : "NO");

  if (correct && benchmark) {
    for (int i = 0; i < 5; ++i) {
      launch();
    }
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
    const double tops = (2.0 * m * n * k) / (ms * 1e9);
    std::printf("time (median, kernel only): %.4f ms\nINT8 TOPS: %.4f\n", ms,
                tops);
  }

  CUDA_CHECK(cudaFree(d_c));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_a));
  return correct;
}

int main() {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU: %s\nCompute capability: %d.%d\n\n", prop.name, prop.major,
              prop.minor);
  if (prop.major < 6 || (prop.major == 6 && prop.minor < 1)) {
    std::fprintf(stderr, "DP4A requires compute capability 6.1 or newer.\n");
    return EXIT_FAILURE;
  }

  std::printf("Correctness tests\n");
  if (!run_case(17, 19, 23, Input::Random, false) ||
      !run_case(17, 19, 23, Input::Zero, false)) {
    return EXIT_FAILURE;
  }
  // Cover K < 4, each scalar-tail length, and packed signed extremes.
  for (int k = 1; k <= 7; ++k) {
    if (!run_case(3, k, 5, Input::Extremes, false)) {
      return EXIT_FAILURE;
    }
  }

  std::printf("\nBenchmarks: 5 warmups, 20 timed launches per shape\n");
  for (int size : {128, 256, 512}) {
    if (!run_case(size, size, size, Input::Random, true)) {
      return EXIT_FAILURE;
    }
  }
  std::printf("\nPASS: all outputs exactly match the CPU reference.\n");
  return EXIT_SUCCESS;
}
