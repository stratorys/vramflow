NVCC := nvcc

ARCH := sm_61

NVCC_FLAGS := \
	-O3 \
	-arch=$(ARCH) \
	-lineinfo

TARGET := dp4a

SRC := src/00_dp4a.cu

all:
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(TARGET)

run: all
	./$(TARGET)

sass: all
	cuobjdump --dump-sass ./$(TARGET)

ptx: all
	cuobjdump --dump-ptx ./$(TARGET)

clean:
	rm -f $(TARGET)
