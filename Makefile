GPU_SM_ARCH=$(MAKECMDGOALS)
GPU_COMPUTE_ARCH=$(subst sm,compute,$(GPU_SM_ARCH))
NVCC=/usr/local/cuda-8.0/bin/nvcc
SRC_DIR=./src/
OBJ_DIR=./obj/
LIB_DIR=./lib/
INCLUDE_DIR=./include/
LOBJS=  gasal.o
LOBJS_PATH=$(addprefix $(OBJ_DIR),$(LOBJS))
VPATH=src:obj:lib

ifeq ($(GPU_SM_ARCH),)
error:
	@echo "Must specify GPU architecture as SM_XX"
endif


ifneq ($(GPU_SM_ARCH),clean)

.SUFFIXES: .cu .c .o .cc .cpp
.cu.o:
	$(NVCC) -c -g -O3 -Xcompiler -Wall -Xptxas -Werror  --gpu-architecture=$(GPU_COMPUTE_ARCH) --gpu-code=$(GPU_SM_ARCH) -lineinfo --ptxas-options=-v --default-stream per-thread $< -o $(OBJ_DIR)$@

$(GPU_SM_ARCH): makedir libgasal.a

makedir:
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(LIB_DIR)
	@mkdir -p $(INCLUDE_DIR)
	@cp $(SRC_DIR)/gasal.h $(INCLUDE_DIR) 
	 

libgasal.a: $(LOBJS)
	ar -csru $(LIB_DIR)$@ $(LOBJS_PATH)
	
endif
	
clean:
	rm -f -r $(OBJ_DIR) $(LIB_DIR) $(INCLUDE_DIR)  *~ *.exe *.o *.txt *~

gasal.o: gasal.h gasal_kernels_inl.h


