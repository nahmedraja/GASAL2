#!/bin/bash
./configure.sh /usr/local/cuda-10.0/
make GPU_SM_ARCH=sm_35 MAX_SEQ_LEN=153 N_CODE=4 N_PENALTY=1
cd test_prog
make $1
sleep 5s
make $12
sleep 5s
cd ..
sha test_prog/golden.log
sha256sum test_prog/out.log
