#!/bin/bash

./personal_configure.sh
make clean
make MAX_SEQ_LEN=150
cd test_prog
make
make fullrun
#cat out.log
sha256sum out.log
cd ..
