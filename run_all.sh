#!/bin/bash
./configure.sh /usr/local/cuda-9.2
make clean
make GPU_SM_ARCH=sm_35 MAX_SEQ_LEN=300 N_CODE=0x4E N_PENALTY=-2
cd test_prog
make $1 
sha256sum -c crc_out.log.crc
cd ..