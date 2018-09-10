#!/bin/bash
./configure.sh /usr/local/cuda/
make clean
make MAX_SEQ_LEN=350
cd test_prog
make
make $1 
#cat out.log
sha256sum -c crc_out.log.crc
cd ..
