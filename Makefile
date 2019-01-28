GPU_SM_ARCH=
MAX_SEQ_LEN=
N_CODE=
N_PENALTY=

GPU_COMPUTE_ARCH=$(subst sm,compute,$(GPU_SM_ARCH))
NVCC=/usr/local/cuda-10.0//bin/nvcc
CC=g++
SRC_DIR=./src/
OBJ_DIR=./obj/
LIB_DIR=./lib/
INCLUDE_DIR=./include/

SOURCES=  args_parser.cpp gasal_align.cu host_batch.cpp ctors.cpp interfaces.cpp res.cpp
LOBJS=$(patsubst %,%o,$(SOURCES))

LOBJS_PATH=$(addprefix $(OBJ_DIR),$(LOBJS))
VPATH=src:obj:lib
YELLOW=\033[1;33m
NC=\033[0m # No Color

ifeq ($(GPU_SM_ARCH),)
error1:
	@echo "Must specify GPU architecture as sm_xx"
endif
ifeq ($(MAX_SEQ_LEN),)
error2:
	@echo "Must specify maximum sequence length"
endif

ifeq ($(N_CODE),)
error3:
	@echo "Must specify the code for 'N'"
endif
#ifneq ($(GPU_SM_ARCH),clean)




## If your computer ships gcc-5.3.1 (at least for CUDA 8.0), this is the regular line. You might need to add: --compiler-options -fPIC 
## With Debian and clang, use: $(NVCC) -ccbin clang-3.8 --compiler-options -fpie

ifeq ($(N_PENALTY),)
%.cuo: %.cu
	$(NVCC) -c -g -O3 -std=c++11 -Xcompiler -Wall,-DMAX_SEQ_LEN=$(MAX_SEQ_LEN),-DN_CODE=$(N_CODE) -Xptxas -Werror  --gpu-architecture=$(GPU_COMPUTE_ARCH) --gpu-code=$(GPU_SM_ARCH) -lineinfo --ptxas-options=-v --default-stream per-thread $< -o $(OBJ_DIR)$@
	
else
%.cuo: %.cu
	$(NVCC) -c -g -O3 -std=c++11 -Xcompiler -Wall,-DMAX_SEQ_LEN=$(MAX_SEQ_LEN),-DN_CODE=$(N_CODE),-DN_PENALTY=$(N_PENALTY) -Xptxas -Werror  --gpu-architecture=$(GPU_COMPUTE_ARCH) --gpu-code=$(GPU_SM_ARCH) -lineinfo --ptxas-options=-v --default-stream per-thread $< -o $(OBJ_DIR)$@
	
endif



## If your computer ships gcc-5.3.1 (at least for CUDA 8.0), this is the regular line. You might need to add: -fPIC 
ifeq ($(N_PENALTY),)
%.cppo: %.cpp
	$(CC) -c -g -O3 -std=c++11 -Wall -DMAX_SEQ_LEN=$(MAX_SEQ_LEN) -DN_CODE=$(N_CODE) -Werror $< -o $(OBJ_DIR)$@
	
else
%.cppo: %.cpp
	$(CC) -c -g -O3 -std=c++11 -Wall -DMAX_SEQ_LEN=$(MAX_SEQ_LEN) -DN_CODE=$(N_CODE) -DN_PENALTY=$(N_PENALTY) -Werror $< -o $(OBJ_DIR)$@
	
endif


all: clean makedir libgasal.a

makedir:
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(LIB_DIR)
	@mkdir -p $(INCLUDE_DIR)
	@cp $(SRC_DIR)/*.h $(INCLUDE_DIR)
	@sed  -i "s/MAX_SEQ_LEN=[0-9]\{1,9\}/MAX_SEQ_LEN=$(MAX_SEQ_LEN)/" ./test_prog/Makefile
	 
ifeq ($(N_PENALTY),)
libgasal.a: $(LOBJS)
	ar -csru $(LIB_DIR)$@ $(LOBJS_PATH)
	@echo ""
	@echo -e "${YELLOW}WARNING:${NC}\"N\" is not defined"
else
libgasal.a: $(LOBJS)
	ar -csru $(LIB_DIR)$@ $(LOBJS_PATH)
endif
	
clean:
	rm -f -r $(OBJ_DIR) $(LIB_DIR) $(INCLUDE_DIR)  *~ *.exe *.cppo *.cuo *.txt *~

gasal_align.cuo: gasal.h gasal_kernels.h


