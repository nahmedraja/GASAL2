CUDA_LD_LIBRARY=/usr/lib/x86_64-linux-gnu/
MAX_SEQ_LEN=300

.cpp.o:
	g++ -g -c -O3 -Wall -DMAX_SEQ_LEN=$(MAX_SEQ_LEN) -Werror -fopenmp -I ../include  -o test_prog.o test_prog.cpp

all: clean test_prog.out

test_prog.out: test_prog.o
	g++ -O3 -o test_prog.out -L$(CUDA_LD_LIBRARY) -L../lib test_prog.o -fopenmp -lcudart -lgasal

clean:
	rm -f -r *~ *.exe *.o *.out *.txt *~ *.tsv
	
test_prog.o: Timer.h

run: all
	optirun ./test_prog.out -p -y semi_global short_query_batch.fasta short_target_batch.fasta

fullrun: all
	optirun ./test_prog.out -p -y semi_global query_batch.fasta target_batch.fasta
