# VRAMFlow

Run models larger than VRAM by overlapping weight streaming with GPU compute.

This CUDA C++ proof of concept targets the GTX 1080 (`sm_61`). Each experiment
answers one question before the next stage starts.

## Build and run

On a Linux NVIDIA machine with Docker and GPU container support:

```sh
docker build -t dp4a-poc .
./scripts/dev.sh
```

Inside the container, or directly on a machine with the CUDA toolkit:

```sh
make all
make run
make run-gemm-naive
make sass-gemm-naive
```

The Docker image uses CUDA 12.2.2. The default build targets `sm_61`; a Mac cannot
run these CUDA experiments. `make clean` removes the generated binaries.

## Experiments

| Stage | Question | Gate |
| --- | --- | --- |
| 00 DP4A | Does native INT8 DP4A work? | Correct output and `IDP.4A` in SASS |
| 01 Naive GEMM | Can DP4A compute a correct INT8 GEMM? | Exact CPU comparison; no throughput threshold |
| 02 Tiled GEMM | Can we feed DP4A efficiently from VRAM? | Compare throughput with cuBLAS INT8 |
| 03 Pinned H2D (planned) | What is the actual RAM → VRAM bandwidth? | Measure 64 MiB through 1 GiB |
| 04 Overlap (planned) | Can compute hide weight transfers? | Compare sequential, double/triple buffering and compute alone |
| 05 LTX block (planned) | Can one real block be reproduced? | Numerical validation against a reference |
| 06 LTX transformer (planned) | Can a full forward fit with streamed weights? | Correctness, peak VRAM and streaming overhead |

The POC stops after stage 06 for evaluation. LTX dimensions and weight sizes must
be verified before implementation; a Rust runtime is a later decision.

## Naive GEMM validation

`src/01_gemm_naive.cu` computes row-major signed INT8 `A[M,K] × B[K,N]` into
INT32 `C[M,N]`. One thread computes one output using signed `__dp4a` calls and
a scalar tail for K values not divisible by four. B remains row-major without
prepacking or transposition.

Running `make run-gemm-naive` checks every output against an INT64 CPU reference:

- Deterministic random and zero inputs at `M=17, K=19, N=23`.
- Signed extremes including `-128` and `127`, with K from 1 through 7.
- Deterministic random inputs in `[-8, 7]` at `128³`, `256³`, and `512³`.

Any mismatch or CUDA error stops the program with a nonzero exit code. Built-in
inputs keep the result within INT32 range. Each square benchmark reports the
median of 20 kernel timings after five warmups. Allocation, transfers and CPU
validation are excluded. TOPS counts multiply and add separately: `2*M*N*K / s`.

Inspect `make sass-gemm-naive` for `IDP.4A` in the `gemm_naive` kernel. An optional
memory check, when Compute Sanitizer is installed, is:

```sh
compute-sanitizer --tool memcheck --error-exitcode 1 ./gemm_naive
```

Record actual GPU output and environment details in [results/GTX1080.md](results/GTX1080.md).

## Tiled GEMM and cuBLAS

`src/02_gemm_tiled.cu` produces a 64×64 output tile per block, with 256 threads
and 16 INT32 accumulators per thread. It stages 32 scalar K values at a time in
shared memory, then uses signed DP4A in registers. A and transposed B are packed
on CPU into four-byte groups along K; the final group is zero-padded. Output
dimensions and K need not be multiples of the tile sizes.

The executable compares two compile-time shared-memory strides: 65 (original
baseline) and 68. For the cooperative store mapping, stride 68 distributes the
32 words of a warp over distinct shared-memory banks; stride 65 can map four
distinct words to the same bank. Both kernels use the same packed inputs,
output buffer, tile dimensions and validation. Each variant is checked against
CPU and, for aligned shapes, cuBLAS. Benchmark output includes both timings,
the stride-68 speedup and each variant's throughput relative to cuBLAS. This
isolates the padding change; its actual throughput benefit must be measured.

```sh
make check-gemm-tiled          # small cases, exhaustive CPU validation
make run-gemm-tiled            # checks followed by the complete benchmark suite
./gemm_tiled --bench           # large benchmarks only
make sass-gemm-tiled           # inspect both specializations for IDP.4A.S8.S8
```

The small tests cover 128³/256³/512³, rectangular and tile-boundary cases, zeros,
and signed extremes with K from 1 through 7. The large suite covers 1024³,
2048³, 4096³, and `(M,K,N)` = `(8192,4096,4096)`, `(8192,4096,16384)`, and
`(8192,16384,4096)`. Each large output is compared exhaustively with cuBLAS and
64 deterministic positions are also checked against an INT64 CPU reference.

The cuBLAS baseline uses `cublasGemmEx`, INT8 inputs, INT32 computation/output,
alpha=1 and beta=0. It computes `Cᵀ = Bᵀ × Aᵀ` using the original row-major
buffers as column-major matrices. Small unaligned cases use the CPU reference
alone; cuBLAS is mandatory for every large benchmark. Any backend error or
numerical mismatch stops the run with a nonzero exit code.

Both backends report the median of 20 CUDA event timings after five warmups.
Allocation, H2D transfers, packing and validation are excluded. CPU packing time
is reported separately; tiled throughput assumes weights are already packed.
The throughput ratio compares GPU compute, not end-to-end inference. There is
no automatic performance pass threshold: review the gap before proceeding to H2D.
Compilation prints register usage and spills via ptxas for subsequent tuning.

Optional sanitizer checks on the small suite:

```sh
compute-sanitizer --tool memcheck --error-exitcode 1 ./gemm_tiled --check
compute-sanitizer --tool racecheck --error-exitcode 1 ./gemm_tiled --check
compute-sanitizer --tool synccheck --error-exitcode 1 ./gemm_tiled --check
```
