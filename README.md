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
| 02 Tiled GEMM (planned) | Can we feed DP4A efficiently from VRAM? | Compare throughput with cuBLAS INT8 |
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
