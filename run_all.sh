#!/bin/bash
./personal_configure.sh
make
cd test_prog
make
make fullrun
cd ..
