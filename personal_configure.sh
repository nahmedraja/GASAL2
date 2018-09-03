#!/bin/bash

export PERL5LIB=.
cuda_path=/usr/lib/nvidia-cuda-toolkit

RED='\033[0;31m'
NC='\033[0m' # No Color

# These paths are valid for a Debian installation of CUDA 8.0 froms its repositories.
cuda_nvcc_path=$cuda_path/bin/nvcc

cuda_lib_path="/usr/lib/x86_64-linux-gnu/"

cuda_runtime_file="/usr/include/cuda_runtime.h"



echo "Configuring Makefile..."

sed  -i "s,NVCC=.*,NVCC=$cuda_nvcc_path,g" Makefile 

echo "Configuring gasal.h..."

sed  -i "s,.*cuda_runtime\.h\",\#include \"$cuda_runtime_file\",g" ./src/gasal.h

echo "Configuring Makefile of test program..."

sed  -i "s,CUDA_LD_LIBRARY=.*,CUDA_LD_LIBRARY=$cuda_lib_path,g" ./test_prog/Makefile 

#mkdir -p include

#cp ./src/gasal.h ./include

echo "Done"


