#!/bin/sh

echo "Log">runner_script.log
start=`date +%R`
echo "began at" $start>>runner_script.log


#==================================== 150 ======================================

STEP=16

i="8"
MAX="153"
OUTPUT_FILE=out150.log
GOLDEN_FILE=golden150.log
MAKE_CMD=human150banded

sed  -i "s/MAX_SEQ_LEN=[0-9]\{1,9\}/MAX_SEQ_LEN=157/g" ./run_all.sh
make GPU_SM_ARCH=sm_35 MAX_SEQ_LEN=$MAX N_CODE=0x4E N_PENALTY=-2

cd test_prog/


echo "BEGIN" > $OUTPUT_FILE
echo "BEGIN" > $GOLDEN_FILE
echo "Running golden sim :"  $(echo $MAKE_CMD | sed s/banded//g)
make $(echo $MAKE_CMD | sed s/banded//g) >> $GOLDEN_FILE 2>&1

while [ $i -lt $MAX ]
do
	sed -i "s/-k [0-9]\{1,9\}/-k $i/g" ./Makefile
	echo "Running banded sim with band " $i
	echo "======================================================" >> out150.log
	make $MAKE_CMD >> $OUTPUT_FILE 2>&1
	i=$[$i+$STEP]
done

cd ..

start=`date +%R`
echo "checkpoint 150 at" $start>>runner_script.log

#==================================== 300 ======================================

STEP=16

i="8"
MAX="305"
OUTPUT_FILE=out300.log
GOLDEN_FILE=golden300.log
MAKE_CMD=human300banded


sed  -i "s/MAX_SEQ_LEN=[0-9]\{1,9\}/MAX_SEQ_LEN=307/g" ./run_all.sh
make GPU_SM_ARCH=sm_35 MAX_SEQ_LEN=$MAX N_CODE=0x4E N_PENALTY=-2

cd test_prog/


echo "BEGIN" > $OUTPUT_FILE
echo "BEGIN" > $GOLDEN_FILE
echo "Running golden sim :"  $(echo $MAKE_CMD | sed s/banded//g)
make $(echo $MAKE_CMD | sed s/banded//g) >> $GOLDEN_FILE 2>&1

while [ $i -lt $MAX ]
do
	sed -i "s/-k [0-9]\{1,9\}/-k $i/g" ./Makefile
	echo "Running banded sim with band " $i
	echo "======================================================" >> out300.log
	make $MAKE_CMD >> $OUTPUT_FILE 2>&1
	i=$[$i+$STEP]
done

cd ..


start=`date +%R`
echo "checkpoint 300 at" $start>>runner_script.log

#==================================== 600 ======================================

STEP=16

i="8"
MAX="601"
OUTPUT_FILE=out600.log
GOLDEN_FILE=golden600.log
MAKE_CMD=human600banded


sed  -i "s/MAX_SEQ_LEN=[0-9]\{1,9\}/MAX_SEQ_LEN=607/g" ./run_all.sh
make GPU_SM_ARCH=sm_35 MAX_SEQ_LEN=$MAX N_CODE=0x4E N_PENALTY=-2

cd test_prog/


echo "BEGIN" > $OUTPUT_FILE
echo "BEGIN" > $GOLDEN_FILE
echo "Running golden sim :"  $(echo $MAKE_CMD | sed s/banded//g)
make $(echo $MAKE_CMD | sed s/banded//g) >> $GOLDEN_FILE 2>&1

while [ $i -lt $MAX ]
do
	sed -i "s/-k [0-9]\{1,9\}/-k $i/g" ./Makefile
	echo "Running banded sim with band " $i
	echo "======================================================" >> out600.log
	make $MAKE_CMD >> $OUTPUT_FILE 2>&1
	i=$[$i+$STEP]
done

cd ..


start=`date +%R`
echo "checkpoint 600 at" $start>>runner_script.log


stop=`date +%R`
echo "stopped - done."
