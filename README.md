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
and 16 INT32 accumulators per thread. Six compile-time kernels compare scalar
K depths 32, 64 and 128 at each shared-memory stride, 65 and 68. Each tile loads
all its inputs, synchronizes, computes signed DP4A in registers, then synchronizes
before shared memory is reused. Larger depths reduce the number of barriers,
but may increase resource pressure; any throughput benefit must be measured.

A and transposed B are packed on CPU into four-byte groups along K; the final
group is zero-padded. Output dimensions and K need not be multiples of the tile
sizes. Cooperative loading preserves the original eight-group mapping in each
512-word slab at every depth. With stride 68, the 32 words stored by a warp map
to distinct shared-memory banks; stride 65 can map four distinct words to the
same bank. The two shared arrays use 4,352 / 8,704 / 17,408 bytes at stride 68
for K depths 32 / 64 / 128, respectively.

All six variants use the same packed inputs, output buffer and validation.
Each is checked against CPU and, for aligned shapes, cuBLAS. Compare depths
at the same stride to isolate the depth change; K=32 is the baseline for each
stride. Benchmark output includes time, TOPS, speedup against that baseline,
and throughput as a percentage of cuBLAS.

```sh
make check-gemm-tiled          # small cases, exhaustive CPU validation
make run-gemm-tiled            # checks followed by the complete benchmark suite
./gemm_tiled --bench           # large benchmarks only
make sass-gemm-tiled           # inspect all six specializations for IDP.4A.S8.S8
```

The small tests cover 128³/256³/512³, rectangular and tile-boundary cases, zeros,
and signed extremes with K from 1 through 7. Additional rectangular random and
extreme-input tests cover K=31/32/33, 63/64/65 and 127/128/129 to exercise
every depth boundary. The large suite covers 1024³,
2048³, 4096³, and `(M,K,N)` = `(8192,4096,4096)`, `(8192,4096,16384)`, and
`(8192,16384,4096)`. Each large output is compared exhaustively with cuBLAS and
64 deterministic positions are also checked against an INT64 CPU reference.

The cuBLAS baseline uses `cublasGemmEx`, INT8 inputs, INT32 computation/output,
alpha=1 and beta=0. It computes `Cᵀ = Bᵀ × Aᵀ` using the original row-major
buffers as column-major matrices. Small unaligned cases use the CPU reference
alone; cuBLAS is mandatory for every large benchmark. Any backend error or
numerical mismatch stops the run with a nonzero exit code.

All backends report the median of 20 CUDA event timings after five warmups.
The benchmark runs three complete campaigns over the same shapes and
deterministic inputs, rotating the order of all seven backends (including
cuBLAS) between campaigns. The order and campaign number are printed; timings
and ratios are reported separately for each campaign.
Allocation, H2D transfers, packing and validation are excluded. CPU packing time
is reported separately; tiled throughput assumes weights are already packed.
The throughput ratio compares GPU compute, not end-to-end inference. There is
no automatic performance pass threshold: review the gap before proceeding to H2D.
Compilation prints register usage, shared memory and spills via ptxas for each
specialization. Archive these alongside the raw campaign output and environment
versions. Retain a deeper tile only if its gain on large shapes reproduces across
campaigns; if differences change sign or remain within observed run-to-run
variation, retain K=32. The stage-02 selection target remains at least 50% of
cuBLAS throughput on representative shapes. No buffering change is included in
this experiment, and a depth speedup alone does not establish that barriers
were the sole bottleneck.

Optional sanitizer checks on the small suite:

```sh
compute-sanitizer --tool memcheck --error-exitcode 1 ./gemm_tiled --check
compute-sanitizer --tool racecheck --error-exitcode 1 ./gemm_tiled --check
compute-sanitizer --tool synccheck --error-exitcode 1 ./gemm_tiled --check
```
