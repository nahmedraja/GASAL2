
all:
	g++ -c -std=c++11 -O3 -Wall -Werror -fopenmp -I ../include  -o test_prog.o test_prog.cpp
	g++ -std=c++11 -O3 -o test_prog.out -L/usr/local/cuda/lib64 -L../lib test_prog.o -fopenmp -lcudart -lgasal

clean:
	rm -f -r *~ *.exe *.o *.out *.txt *~ *.tsv
	
