CUDA_LD_LIBRARY=/usr/lib/x86_64-linux-gnu/
ANALYSIS_FILENAME=analysis

.cpp.o:
	g++ -g -c -O3 -Wall -Werror -fopenmp -I ../include  -o test_prog.o test_prog.cpp

all: clean test_prog.out

test_prog.out: test_prog.o
	g++ -O3 -o test_prog.out -L$(CUDA_LD_LIBRARY) -L../lib test_prog.o -fopenmp -lcudart -lgasal

clean:
	rm -f -r *~ *.exe *.o *.out *.txt *~ *.tsv
	
test_prog.o: Timer.h

run: all
	optirun ./test_prog.out -p -y semi_global short_query_batch.fasta short_target_batch.fasta  

fullrun: all
	optirun ./test_prog.out -p -y semi_global query_batch.fasta target_batch.fasta > out.log

nvprof: all
	optirun nvprof -f -o $(ANALYSIS_FILENAME).nvprof ./test_prog.out -p -y semi_global query_batch_rc.fasta target_batch_rc.fasta > out.log
	nvvp $(ANALYSIS_FILENAME).nvprof

nvprof-262k: all
	optirun nvprof -f -o $(ANALYSIS_FILENAME).nvprof ./test_prog.out -p -y semi_global reads_150.fasta ref_150.fasta > out.log
	nvvp $(ANALYSIS_FILENAME).nvprof

262k: all
	optirun ./test_prog.out -p -y semi_global reads_150.fasta ref_150.fasta > out.log

cuda-memcheck: all
	optirun cuda-memcheck ./test_prog.out -p -y semi_global query_batch.fasta target_batch.fasta

cuda-gdb: all
	optirun cuda-gdb --args ./test_prog.out -p -y semi_global query_batch.fasta target_batch.fasta

valgrind: all
	optirun valgrind ./test_prog.out -p -y semi_global short_query_batch.fasta short_target_batch.fasta

gdb: all
	optirun gdb --args ./test_prog.out -p -y semi_global short_query_batch.fasta short_target_batch.fasta
