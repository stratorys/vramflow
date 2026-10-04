NVCC := nvcc

ARCH := sm_61

NVCC_FLAGS := \
	-O3 \
	-std=c++14 \
	-arch=$(ARCH) \
	-lineinfo

TARGETS := dp4a gemm_naive

.PHONY: all run sass ptx run-gemm-naive sass-gemm-naive clean

all: $(TARGETS)

dp4a: src/00_dp4a.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

gemm_naive: src/01_gemm_naive.cu src/common.cuh
	$(NVCC) $(NVCC_FLAGS) $< -o $@

run: dp4a
	./dp4a

sass: dp4a
	cuobjdump --dump-sass ./dp4a

ptx: dp4a
	cuobjdump --dump-ptx ./dp4a

run-gemm-naive: gemm_naive
	./gemm_naive

sass-gemm-naive: gemm_naive
	cuobjdump --dump-sass ./gemm_naive

clean:
	rm -f $(TARGETS)
