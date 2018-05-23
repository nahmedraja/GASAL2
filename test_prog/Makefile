
CUDA_LD_LIBRARY=/usr/local/cuda-9.2/targets/x86_64-linux/lib
MAX_SEQ_LEN=600

.cpp.o:
	g++ -g -c -O3 -Wall -DMAX_SEQ_LEN=600

all: test_prog.out

test_prog.out: test_prog.o
	g++ -O3 -o test_prog.out -L$(CUDA_LD_LIBRARY) -L../lib test_prog.o -fopenmp -lcudart -lgasal

clean:
	rm -f -r *~ *.exe *.o *.out *.txt *~ *.tsv
	
test_prog.o: Timer.h
