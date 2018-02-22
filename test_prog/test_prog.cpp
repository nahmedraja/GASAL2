#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <unistd.h>
#include <cstdlib>
#include <stdint.h>
#include <math.h>
#include <omp.h>
#include "Timer.h"


#include "gasal.h"

using namespace std;


#define BATCH_SIZE 256*1024



int main(int argc, char *argv[]) {
	int32_t c, sa = 1, sb = 4;
	int32_t gapo = 6, gape = 1, minsc = 0;
	int start_pos = 0;
	int print_out = 0;
	int n_threads = 1;
	std::string al_type;

// parse command line
	while ((c = getopt(argc, argv, "a:b:q:r:t:n:y:sp")) >= 0) {
		switch (c) {
		case 'a':
			sa = atoi(optarg);
			break;
		case 'b':
			sb = atoi(optarg);
			break;
		case 'q':
			gapo = atoi(optarg);
			break;
		case 'r':
			gape = atoi(optarg);
			break;
		case 't':
			minsc = atoi(optarg);
			break;
		case 's':
			start_pos = 1;
			break;
		case 'p':
			print_out = 1;
			break;
		case 'n':
			n_threads = atoi(optarg);
			break;
		case 'y':
			al_type = std::string(optarg);
			break;

		}
	}

	if (optind + 2 > argc) {
		fprintf(stderr, "Usage: ksw [-a] [-b] [-q] [-r] [-t] [-o] [-g] <batch1.fasta> <batch2.fasta>\n");
		fprintf(stderr, "Options: -a INT    match score [%d]\n", sa);
		fprintf(stderr, "         -b INT    mismatch penalty [%d]\n", sb);
		fprintf(stderr, "         -q INT    gap open penalty [%d]\n", gapo);
		fprintf(stderr, "         -r INT    gap extension penalty [%d]\n", gape);
		fprintf(stderr, "         -t INT    scores >= INT are only reported [%ld]\n", (long) minsc);
		fprintf(stderr, "         -t INT    scores >= INT are only reported [%ld]\n", (long) minsc);
		fprintf(stderr, "         -s        also find the start position \n");
		fprintf(stderr, "         -p        print the output \n");
		fprintf(stderr, "         -n        Number of threads \n");
		fprintf(stderr, "         -y        Alignment type (Must be either \"local\" or \"semi_global\") \n");
		fprintf(stderr, "\n");
		return 1;
	}
	if (al_type.empty()) {
		fprintf(stderr, "Must specify the alignment type (local, semi_global)\n");
		return 1;

	}
	if ( al_type.compare("local") != 0 && al_type.compare("semi_global") != 0 && al_type.compare("global") != 0) {
		fprintf(stderr, "Unknown alignment type. Must be either \"local\" or \"semi_global\" or \"global\")\n");
		return 1;
	}

	gasal_subst_scores sub_scores;

	sub_scores.match = sa;
	sub_scores.mismatch = sb;
	sub_scores.gap_open = gapo;
	sub_scores.gap_extend = gape;

	gasal_copy_subst_scores(&sub_scores);



	ifstream batch1_fasta(argv[optind]);
	ifstream batch2_fasta(argv[optind + 1]);

	vector<string> batch1_vec;
	vector<string> batch2_vec;
	vector<string> batch1_header;
	vector<string> batch2_header;
	string batch1_line, batch2_line;

	int total_seqs = 0;
	cerr << "Loading files...." << endl;
	while (getline(batch1_fasta, batch1_line) && getline(batch2_fasta, batch2_line)) {
		if (batch1_line[0] == '>' && batch2_line[0] == '>') {
			batch1_header.push_back(batch1_line.substr(1));
			batch2_header.push_back(batch2_line.substr(1));
			getline(batch1_fasta, batch1_line);
			batch1_vec.push_back(batch1_line);
			getline(batch2_fasta, batch2_line);
			batch2_vec.push_back(batch2_line);
			total_seqs++;
		} else {
			cerr << "Batch1 and batch2 files should be fasta having same number of sequences" << endl;
			exit(EXIT_FAILURE);
		}

	}

	int *thread_idx = (int*)malloc(n_threads*sizeof(int));
	int *thread_n_seqs = (int*)malloc(n_threads*sizeof(int));
	//int *total_thread_n_seqs = (int*)calloc(n_threads, sizeof(int));
	double *thread_misc_time = (double*)calloc(n_threads, sizeof(double));

	int thread_batch_size = (int)ceil((double)BATCH_SIZE/*/n_threads*/);
	if(thread_batch_size%128) thread_batch_size = thread_batch_size + (128 - (thread_batch_size%128));
	int seqs_done = 0;
	cerr << "Processing..." << endl;

	Timer total_time;
	total_time.Start();
	while (seqs_done < total_seqs) {
		for (int i = 0; i < n_threads; i++){
			if (seqs_done + thread_batch_size < total_seqs)
				thread_n_seqs[i] = thread_batch_size;
			else
				thread_n_seqs[i] = total_seqs - seqs_done;
			thread_idx[i] = seqs_done;
			seqs_done += thread_n_seqs[i];
		}
		omp_set_num_threads(n_threads);
#pragma omp parallel
		{
			int n_seqs = thread_n_seqs[omp_get_thread_num()];
			if (n_seqs > 0) {
				int curr_idx = thread_idx[omp_get_thread_num()];

				vector<uint8_t> batch1(304*BATCH_SIZE);
				vector<uint8_t> batch2(600*BATCH_SIZE);
				vector<uint32_t> batch1_offsets(n_seqs);
				vector<uint32_t> batch2_offsets(n_seqs);
				vector<uint32_t> batch1_lens(n_seqs);
				vector<uint32_t> batch2_lens(n_seqs);

				int batch_idx = 0;
				Timer misc_time;
				misc_time.Start();
				for (int i = curr_idx, j = 0; i < n_seqs + curr_idx; i++, j++) {
					memcpy(&(batch1[batch_idx]), batch1_vec[i].c_str(), batch1_vec[i].size());
					batch1_offsets[j] = batch_idx;
					batch_idx += batch1_vec[i].size();
					int seq_len = batch1_vec[i].size();
					while(batch_idx%8) {
						batch1[batch_idx++] = 'P';
					}
					batch1_lens[j] = seq_len;

				}
				uint32_t batch1_bytes = batch_idx;

				batch_idx = 0;
				for (int i = curr_idx, j = 0; i < n_seqs + curr_idx; i++, j++) {
					memcpy(&(batch2[batch_idx]), batch2_vec[i].c_str(), batch2_vec[i].size());
					batch2_offsets[j] = batch_idx;
					batch_idx += batch2_vec[i].size();
					int seq_len = batch2_vec[i].size();
					while(batch_idx%8) {
						batch2[batch_idx++] = 'P';
					}
					batch2_lens[j] = seq_len;

				}
				uint32_t batch2_bytes = batch_idx;
				misc_time.Stop();
				thread_misc_time[omp_get_thread_num()] += misc_time.GetTime();
				//gasal_gpu_storage *gpu_storage = NULL;

				vector<int32_t> scores, batch1_start, batch2_start, batch1_end, batch2_end;
				scores.resize(n_seqs);
				if (al_type.compare("local") == 0){
					batch1_end.resize(n_seqs);
					batch2_end.resize(n_seqs);
					if(start_pos){
						batch1_start.resize(n_seqs);
						batch2_start.resize(n_seqs);
					}
				}
				else if (al_type.compare("semi_global") == 0) {
					batch2_end.resize(n_seqs);
					if (start_pos) {
						batch2_start.resize(n_seqs);
					}
				}

				if (al_type.compare("local") == 0){
					if(start_pos){
						gasal_aln(batch1.data(), batch1_offsets.data(), batch1_lens.data(), batch2.data(), batch2_offsets.data(), batch2_lens.data(), batch1_bytes, batch2_bytes, n_seqs, scores.data(), batch1_start.data(), batch2_start.data(), batch1_end.data(), batch2_end.data(), LOCAL, WITH_START);
					}
					else {
						gasal_aln(batch1.data(), batch1_offsets.data(), batch1_lens.data(), batch2.data(), batch2_offsets.data(), batch2_lens.data(), batch1_bytes, batch2_bytes, n_seqs, scores.data(), NULL, NULL, batch1_end.data(), batch2_end.data(), LOCAL, WITHOUT_START);
					}
				}
				else if (al_type.compare("semi_global") == 0) {
					if (start_pos) {


					} else {
						gasal_aln(batch1.data(), batch1_offsets.data(), batch1_lens.data(), batch2.data(), batch2_offsets.data(), batch2_lens.data(), batch1_bytes, batch2_bytes, n_seqs, scores.data(), NULL, NULL, NULL, batch2_end.data(), SEMI_GLOBAL, WITHOUT_START);

					}
				} else {
					    gasal_aln(batch1.data(), batch1_offsets.data(), batch1_lens.data(), batch2.data(), batch2_offsets.data(), batch2_lens.data(), batch1_bytes, batch2_bytes, n_seqs, scores.data(), NULL, NULL, NULL, batch2_end.data(), SEMI_GLOBAL, WITHOUT_START);

				}


//				if (al_type.compare("local") == 0){
//					if(start_pos){
//						gasal_error_t err = gasal_get_aln_results(gpu_storage, n_seqs, scores.data(), batch1_start.data(), batch2_start.data(), batch1_end.data(), batch2_end.data());
//						while (err != 0) {
//							//cerr << "I am stuck here" << endl;
//							err = gasal_get_aln_results(gpu_storage, n_seqs, scores.data(), batch1_start.data(), batch2_start.data(), batch1_end.data(), batch2_end.data());
//						}
//					} else {
//						gasal_error_t err = gasal_get_aln_results(gpu_storage, n_seqs, scores.data(), NULL, NULL, batch1_end.data(), batch2_end.data());
//						while (err != 0) {
//							//cerr << "I am stuck here" << endl;
//							err = gasal_get_aln_results(gpu_storage, n_seqs, scores.data(), NULL, NULL, batch1_end.data(), batch2_end.data());
//						}
//					}
//				}
//				else if (al_type.compare("semi_global") == 0) {
//					if (start_pos) {
//
//					} else {
//
//					}
//				} else {
//
//				}
				misc_time.Start();
				if(print_out) {
#pragma omp critical
					for (int i = curr_idx, j = 0; i < n_seqs + curr_idx; i++, j++) {
						if(al_type.compare("local") == 0) {
							if (start_pos){
								fprintf(stdout, "seq_set=%s\tscore=%d\tbatch1_start=%d\tbatch2_start=%d\tbatch1_end=%d\tbatch2_end=%d\n", batch1_header[i].c_str(), scores[j], batch1_start[j],
										batch2_start[j], batch1_end[j], batch2_end[j]);
							}
							else {
								fprintf(stdout, "seq_set=%s\tscore=%d\tbatch1_end=%d\tbatch2_end=%d\n", batch1_header[i].c_str(), scores[j], batch1_end[j], batch2_end[j]);
							}
						} else if(al_type.compare("semi_global") == 0) {
							if (start_pos){
								fprintf(stdout, "seq_set=%s\tscore=%d\tbatch2_start=%d\tbatch2_end=%d\n", batch1_header[i].c_str(), scores[j], batch2_start[j], batch2_end[j]);

							}
							else {
								fprintf(stdout, "seq_set=%s\tscore=%d\tbatch2_end=%d\n", batch1_header[i].c_str(), scores[j],batch2_end[j]);
							}
						}   else{
								fprintf(stdout, "seq_set=%s\tscore=%d\n", batch1_header[i].c_str(), scores[j]);
						}
					}
				}
				misc_time.Stop();
				thread_misc_time[omp_get_thread_num()] += misc_time.GetTime();

			}
		}


	}
	total_time.Stop();
	string algo = al_type;
	string start_type[2] = {"without_start", "with_start"};
	al_type += "_";
	al_type += start_type[start_pos];
	double av_misc_time = 0.0;
//	for (int i = 0; i < n_threads; ++i){
//		fprintf(stderr, "Thread[%d] performed  %d %s alignments in %.3f milliseconds\n", i, total_thread_n_seqs[i], al_type.c_str(), thread_aln_time[i]);
//		total_aln_time +=  thread_aln_time[i];
//	}
	for (int i = 0; i < n_threads; ++i){
			av_misc_time += (thread_misc_time[i]/n_threads);
	}
	fprintf(stderr, "\n-------------------------------------------------------------------------------\n");
	fprintf(stderr, "%d threads performed %d %s alignments in %.3f milliseconds\n", n_threads, total_seqs, al_type.c_str(), total_time.GetTime() - av_misc_time);
}
