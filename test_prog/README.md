This directory conatins a test program for GASAL. First compile GASAL with `N_CODE=0x4E`. To compile the test program run `make`. Running the test program without any paramters will print the options:

```
$./test_prog.out

Usage: ./test_prog.out [-a] [-b] [-q] [-r] [-s] [-p] [-n] [-y] <query_batch.fasta> <target_batch.fasta>
Options: -a INT    match score [1]
         -b INT    mismatch penalty [4]
         -q INT    gap open penalty [6]
         -r INT    gap extension penalty [1]
         -s        also find the start position 
         -p        print the alignment results 
         -n        Number of threads 
         -y        Alignment type . Must be "local", "semi_global" or "global"  
```


`query_batch.fasta` and `target_batch.fasta` contain the sequences for the alignment. The sequences in these files are aligned one-to-one, i.e. the first sequence in query_batch.fasta is aligned to the first sequence in target_batch.fasta, the second sequence in query_batch.fasta is aligned to the second sequence in target_batch.fasta, and so on. The directory also conatins sample query_batch.fasta and target_batch.fasta files.

