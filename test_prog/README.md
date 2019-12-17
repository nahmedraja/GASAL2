This directory conatins a test program for GASAL2. The program overlaps the sequence alignment on the GPU with CPU execution. The CPU executes the code for creating a batch of sequences to be aligned on the GPU and printing the alignment results. First compile GASAL with `N_CODE=0x4E`. To compile the test program run `make`. Running the test program with `-h` or `--help` will print the options:

```
$./test_prog.out -h

Usage: ./test_prog.out [-a] [-b] [-q] [-r] [-s] [-p] [-n] [-y] <query_batch.fasta> <target_batch.fasta>
Options: -a INT    match score [1]
         -b INT    mismatch penalty [4]
         -q INT    gap open penalty [6]
         -r INT    gap extension penalty [1]
         -s        also find the start position
	 -t        compute traceback. With this option enabled, "-s" has no effect as start position will always be computed with traceback
         -p        print the alignment results
         -n INT    Number of threads [1]
         -y AL_TYPE       Alignment type . Must be "local", "semi_global", "global", "ksw"
         -x HEAD TAIL     specifies, for semi-global alignment, wha should be skipped for heads and tails of the sequences. (NONE, QUERY, TARGET, BOTH)
         -k INT    Band width in case "banded" is selected.
         --help, -h : displays this message.
         --second-best   displays second best score (WITHOUT_START only).
Single-pack multi-Parameters (e.g. -sp) is not supported.

```


`query_batch.fasta` and `target_batch.fasta` contain the single-line FASTA sequences for the alignment. The sequences in these files are aligned one-to-one, i.e. the first sequence in query_batch.fasta is aligned to the first sequence in target_batch.fasta, the second sequence in query_batch.fasta is aligned to the second sequence in target_batch.fasta, and so on. The directory also contains sample query_batch.fasta and target_batch.fasta files. For the two sample files use `MAX_QUERY_LEN=160`.

In order to demonstrate easily the possibilities of reverse-complementing independently, one can change the first character of the sequence delimiter `>` in the .fasta files. The test program parses the first character as the following :

- Parsing `>` does no operation on the sequence (this is the regular mode),
- Parsing `<` flags the sequence to be reversed,
- Parsing `/` flags the sequence to be complemented,
- Parsing `+` flags the sequence to be reversed and complemented.



