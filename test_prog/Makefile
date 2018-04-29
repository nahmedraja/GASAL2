
CUDA_LD_LIBRARY=/usr/local/cuda-8.0/lib64
MAX_LEN=600

.cpp.o:
	g++ -g -c -O3 -Wall -DMAX_SEQ_LEN=$(MAX_LEN) -Werror -fopenmp -I ../include  -o test_prog.o test_prog.cpp

all: test_prog.out

test_prog.out: test_prog.o
	g++ -O3 -o test_prog.out -L$(CUDA_LD_LIBRARY) -L../lib test_prog.o -fopenmp -lcudart -lgasal

clean:
	rm -f -r *~ *.exe *.o *.out *.txt *~ *.tsv
	
test_prog.o: Timer.h