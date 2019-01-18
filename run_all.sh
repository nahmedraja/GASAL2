#!/bin/bash
./configure.sh /usr/local/cuda-9.2
make GPU_SM_ARCH=sm_50 MAX_SEQ_LEN=300 N_CODE=4 N_PENALTY=1
cd test_prog
make $1 
cd ..
sha256sum -c test_prog/crc_out.log.crc

