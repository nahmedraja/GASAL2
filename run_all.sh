#!/bin/bash
./personal_configure.sh
make clean
make GPU_SM_ARCH=sm_20 MAX_SEQ_LEN=350 N_CODE=0x4E N_PENALTY=2
cd test_prog
make
make $1 
#cat out.log
sha256sum -c crc_out.log.crc
cd ..
