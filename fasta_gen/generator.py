#!/usr/bin/python
import sys 

import random
def main(fixed_length, varia_before, varia_after, nbr_seqs, proba_muta):
	
	
	values=["A", "C", "G", "T"]
	query=open("query_batch.fasta", "w+")
	target=open("target_batch.fasta", "w+")
	completion = 0
	
	for line in range (nbr_seqs):
		query.write(">"+str(line)+"\n")
		target.write(">"+str(line)+"\n")
		r1 = random.random()
		r2 = random.random()
		for char in range(int(r1*varia_before)):
			target.write(random.choice(values))

		for char in range(fixed_length):
			val=random.choice(values)	
			query.write(val)
			mutation = random.random()
			if mutation < proba_muta:
				target.write(val)
			elif mutation >= proba_muta and mutation < 2*proba_muta:
				target.write("")
			elif mutation >= proba_muta and mutation < proba_muta:
				#it's a insertion
				target.write(val)
				target.write(random.choice(values))
			elif mutation >= 3*proba_muta and mutation < 4*proba_muta:
				#it's a modification of the nucleotide
				target.write(random.choice(values))

		for char in range(int(r2*varia_after)):
			target.write(random.choice(values))

		query.write("\n")
		target.write("\n")
		completion = 100*line/nbr_seqs
		sys.stdout.write("Completion: %d%%   \r" % (completion) )
		sys.stdout.flush()
	
	query.close();
	target.close();




main(500, 30, 30, 200000, 0.92)
