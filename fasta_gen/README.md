# Random FASTA files generator

This is a python (2.7) script to generate random FASTA files.

## Behaviour

The script generates two files:

- On one hand, it creates a random `query_batch.fasta`,
- On the other hand, it creates a `targer_batch.fasta` that can have:
	* more bases in the beginning of the sequences, (see next section for parameters),
	* more bases in the end of the sequences,
	* random insertions, deletions, or mutations of bases inside the sequence.

## Parameters

You can edit the parameters of the main function inside the script, on the last line. These parameters are, in order:

- fixed length: the fixed lengths of your sequences read for the query file
- variability before: the maximum amount of random sequences that can show up in the beginning of the sequence (think of this as random padding). For each line, the size of the variability before is randomized, ranging from 0 to the specified number. 
- variability after: same for the tail of the sequence.
- number of sequences
- coefficient of similarity: the more this number is close to 1, the less probable the insersions, deletions or mutations become. Setting this coefficient outside of the range [0,1] leads to unpredicted behaviour.

