#!/bin/bash
./personal_configure.sh
make clean
make MAX_SEQ_LEN=150
cd test_prog
make
make $1 
#cat out.log
sha256sum -c crc_out.log.crc
cd ..
