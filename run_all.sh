#!/bin/bash
# run th" configure.sh script with the install path of CUDA. An example is show below.
./configure.sh /usr/local/cuda-8.0/
# run the makefile with the parameters specified. See Readme for their explanations. An example is shown below.
make GPU_SM_ARCH=sm_35 MAX_SEQ_LEN=300 N_CODE=0x4E N_PENALTY=1

# you can uncomment the following lines to run the test program using the test program makefile rules. 
# By specifying the test program makefile rule as command line argument when running this script.

# cd test_prog
# make $1 
# sha256sum -c crc_out.log.crc
# cd ..
