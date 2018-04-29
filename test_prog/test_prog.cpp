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


#include "../include/gasal.h"

using namespace std;


#define GPU_BATCH_SIZE 10000



int main(int argc, char *argv[]) {
	int32_t c, sa = 1, sb = 4;
	int32_t gapo = 6, gape = 1;
	int start_pos = 0;
	int print_out = 0;
	int n_threads = 1;
	std::string al_type;

// parse command line
	while ((c = getopt(argc, argv, "a:b:q:r:n:y:sp")) >= 0) {
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
		fprintf(stderr, "Usage: ./test_prog.out [-a] [-b] [-q] [-r] [-s] [-p] [-n] [-y] <query_batch.fasta> <target_batch.fasta>\n");
		fprintf(stderr, "Options: -a INT    match score [%d]\n", sa);
		fprintf(stderr, "         -b INT    mismatch penalty [%d]\n", sb);
		fprintf(stderr, "         -q INT    gap open penalty [%d]\n", gapo);
		fprintf(stderr, "         -r INT    gap extension penalty [%d]\n", gape);
		fprintf(stderr, "         -s        also find the start position \n");
		fprintf(stderr, "         -p        print the alignment results \n");
		fprintf(stderr, "         -n        Number of threads \n");
		fprintf(stderr, "         -y        Alignment type . Must be \"local\", \"semi_global\" or \"global\"  \n");
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

	//--------------copy substitution scores to GPU--------------------
	gasal_subst_scores sub_scores;

	sub_scores.match = sa;
	sub_scores.mismatch = sb;
	sub_scores.gap_open = gapo;
	sub_scores.gap_extend = gape;

	gasal_copy_subst_scores(&sub_scores);

	//-------------------------------------------------------------------


	ifstream query_batch_fasta(argv[optind]);
	ifstream target_batch_fasta(argv[optind + 1]);

	vector<string> query_seqs;
	vector<string> target_seqs;
	vector<string> query_headers;
	vector<string> target_headers;
	string query_batch_line, target_batch_line;

	int total_seqs = 0;
	cerr << "Loading files...." << endl;
	while (getline(query_batch_fasta, query_batch_line) && getline(target_batch_fasta, target_batch_line)) { //load sequences from the files
		if (query_batch_line[0] == '>' && target_batch_line[0] == '>') {
			query_headers.push_back(query_batch_line.substr(1));
			target_headers.push_back(target_batch_line.substr(1));
			getline(query_batch_fasta, query_batch_line);
			query_seqs.push_back(query_batch_line);
			getline(target_batch_fasta, target_batch_line);
			target_seqs.push_back(target_batch_line);
			total_seqs++;
		} else {
			cerr << "Batch1 and target_batch files should be fasta having same number of sequences" << endl;
			exit(EXIT_FAILURE);
		}

	}

	int *thread_seqs_idx = (int*)malloc(n_threads*sizeof(int));
	int *thread_n_seqs = (int*)malloc(n_threads*sizeof(int));
	double *thread_misc_time = (double*)calloc(n_threads, sizeof(double));

	int thread_batch_size = (int)ceil((double)total_seqs/n_threads);
	int n_seqs_alloc = 0;
	for (int i = 0; i < n_threads; i++){//distribute the sequences among the threads equally
		thread_seqs_idx[i] = n_seqs_alloc;
		if (n_seqs_alloc + thread_batch_size < total_seqs) thread_n_seqs[i] = thread_batch_size;
		else thread_n_seqs[i] = total_seqs - n_seqs_alloc;
		n_seqs_alloc += thread_n_seqs[i];

	}
	cerr << "Processing..." << endl;


	Timer total_time;
	total_time.Start();
	omp_set_num_threads(n_threads);
	gasal_gpu_storage_t *gpu_storage_arr =(gasal_gpu_storage_t*)calloc(n_threads, sizeof(gasal_gpu_storage_t));
	int z;
	for (z = 0; z < n_threads; z++) {//allocate GPU memory for all threads
		gasal_gpu_mem_alloc(&(gpu_storage_arr[z]), MAX_SEQ_LEN*GPU_BATCH_SIZE, MAX_SEQ_LEN*GPU_BATCH_SIZE, GPU_BATCH_SIZE, LOCAL, WITH_START);
	}
	#pragma omp parallel
	{
	int n_seqs = thread_n_seqs[omp_get_thread_num()];//number of sequences allocated to this thread
	int curr_idx = thread_seqs_idx[omp_get_thread_num()];//number of sequences allocated to this thread
	int seqs_done = 0;
	if (n_seqs > 0) {
		while (seqs_done < n_seqs) {
			vector<uint8_t> query_batch(MAX_SEQ_LEN*GPU_BATCH_SIZE);
			vector<uint8_t> target_batch(MAX_SEQ_LEN*GPU_BATCH_SIZE);
			vector<uint32_t> query_batch_offsets(GPU_BATCH_SIZE);
			vector<uint32_t> target_batch_offsets(GPU_BATCH_SIZE);
			vector<uint32_t> query_batch_lens(GPU_BATCH_SIZE);
			vector<uint32_t> target_batch_lens(GPU_BATCH_SIZE);

			int query_batch_idx = 0;
			int target_batch_idx = 0;
			Timer misc_time;
			misc_time.Start();
			int j = 0;
			//-----------Fill batches with sequences. Pick GPU_BATCH_SIZE number of sequences each time to align on the GPU-----------------------
			for (int i = curr_idx; seqs_done < n_seqs && j < GPU_BATCH_SIZE; i++, j++, seqs_done++) {
				memcpy(&(query_batch[query_batch_idx]), query_seqs[i].c_str(), target_seqs[i].size());
				memcpy(&(target_batch[target_batch_idx]), target_seqs[i].c_str(),  target_seqs[i].size());
				query_batch_offsets[j] = query_batch_idx;
				target_batch_offsets[j] = target_batch_idx;
				query_batch_idx += query_seqs[i].size();
				target_batch_idx +=  target_seqs[i].size();
				int query_batch_seq_len = query_seqs[i].size();
				while(query_batch_idx%8) {
					query_batch[query_batch_idx++] = 'N';
				}
				query_batch_lens[j] = query_batch_seq_len;
				int target_batch_seq_len =  target_seqs[i].size();
				while(target_batch_idx%8) {
					target_batch[target_batch_idx++] = 'N';
				}
				target_batch_lens[j] = target_batch_seq_len;


			}
			int actual_n_seqs = j;
			uint32_t query_batch_bytes = query_batch_idx;
			uint32_t target_batch_bytes = target_batch_idx;

			//----------------------------------------------------------------------------------------------------
			misc_time.Stop();
			thread_misc_time[omp_get_thread_num()] += misc_time.GetTime();
			//gasal_gpu_storage *gpu_storage = NULL;

			vector<int32_t> scores, query_batch_start, target_batch_start, query_batch_end, target_batch_end;//alignment result arrays on the CPU side
			scores.resize(actual_n_seqs);
			if (al_type.compare("local") == 0){
				query_batch_end.resize(actual_n_seqs);
				target_batch_end.resize(actual_n_seqs);
				if(start_pos){
					query_batch_start.resize(actual_n_seqs);
					target_batch_start.resize(actual_n_seqs);
				}
			}
			else if (al_type.compare("semi_global") == 0) {
				target_batch_end.resize(actual_n_seqs);
				if (start_pos) {
					target_batch_start.resize(actual_n_seqs);
				}
			}
			//------------------call the GASAL alignment function---------------------
			if (al_type.compare("local") == 0){
				if(start_pos){
					gasal_aln(&(gpu_storage_arr[omp_get_thread_num()]), query_batch.data(), query_batch_offsets.data(), query_batch_lens.data(), target_batch.data(), target_batch_offsets.data(), target_batch_lens.data(), query_batch_bytes, target_batch_bytes, actual_n_seqs, scores.data(), query_batch_start.data(), target_batch_start.data(), query_batch_end.data(), target_batch_end.data(), LOCAL, WITH_START);

				}
				else {
					gasal_aln(&(gpu_storage_arr[omp_get_thread_num()]), query_batch.data(), query_batch_offsets.data(), query_batch_lens.data(), target_batch.data(), target_batch_offsets.data(), target_batch_lens.data(), query_batch_bytes, target_batch_bytes, actual_n_seqs, scores.data(), NULL, NULL, query_batch_end.data(), target_batch_end.data(), LOCAL, WITHOUT_START);
				}
			}
			else if (al_type.compare("semi_global") == 0) {
				if (start_pos) {
					gasal_aln(&(gpu_storage_arr[omp_get_thread_num()]), query_batch.data(), query_batch_offsets.data(), query_batch_lens.data(), target_batch.data(), target_batch_offsets.data(), target_batch_lens.data(), query_batch_bytes, target_batch_bytes, actual_n_seqs, scores.data(), NULL, target_batch_start.data(), NULL, target_batch_end.data(), SEMI_GLOBAL, WITH_START);

				} else {
					gasal_aln(&(gpu_storage_arr[omp_get_thread_num()]), query_batch.data(), query_batch_offsets.data(), query_batch_lens.data(), target_batch.data(), target_batch_offsets.data(), target_batch_lens.data(), query_batch_bytes, target_batch_bytes, actual_n_seqs, scores.data(), NULL, NULL, NULL, target_batch_end.data(), SEMI_GLOBAL, WITHOUT_START);

				}
			} else {
				gasal_aln(&(gpu_storage_arr[omp_get_thread_num()]), query_batch.data(), query_batch_offsets.data(), query_batch_lens.data(), target_batch.data(), target_batch_offsets.data(), target_batch_lens.data(), query_batch_bytes, target_batch_bytes, actual_n_seqs, scores.data(), NULL, NULL, NULL, NULL, GLOBAL, WITHOUT_START);

			}
			//---------------------------------------------------------------------------------

			misc_time.Start();

			//-------------------------------print alignment results----------------------------------------
			if(print_out) {
#pragma omp critical
				j = 0;
				for (int i = curr_idx; j < actual_n_seqs; i++, j++) {
					if(al_type.compare("local") == 0) {
						if (start_pos){
							fprintf(stdout, "query_name=%s\ttarget_name=%s\tscore=%d\tquery_batch_start=%d\ttarget_batch_start=%d\tquery_batch_end=%d\ttarget_batch_end=%d\n", query_headers[i].c_str(), target_headers[i].c_str(), scores[j], query_batch_start[j],
									target_batch_start[j], query_batch_end[j], target_batch_end[j]);
						}
						else {
							fprintf(stdout, "query_name=%s\ttarget_name=%s\tscore=%d\tquery_batch_end=%d\ttarget_batch_end=%d\n", query_headers[i].c_str(), target_headers[i].c_str(), scores[j], query_batch_end[j], target_batch_end[j]);
						}
					} else if(al_type.compare("semi_global") == 0) {
						if (start_pos){
							fprintf(stdout, "query_name=%s\ttarget_name=%s\tscore=%d\ttarget_batch_start=%d\ttarget_batch_end=%d\n", query_headers[i].c_str(), target_headers[i].c_str(), scores[j], target_batch_start[j], target_batch_end[j]);

						}
						else {
							fprintf(stdout, "query_name=%s\ttarget_name=%s\tscore=%d\ttarget_batch_end=%d\n", query_headers[i].c_str(), target_headers[i].c_str(), scores[j],target_batch_end[j]);
						}
					}   else{
						fprintf(stdout, "query_name=%s\ttarget_name=%s\tscore=%d\n", query_headers[i].c_str(), target_headers[i].c_str(), scores[j]);
					}
				}
			}
			//----------------------------------------------------------------------------------------------------
			misc_time.Stop();
			curr_idx += GPU_BATCH_SIZE;
			thread_misc_time[omp_get_thread_num()] += misc_time.GetTime();

		}
	}


	}
	for (z = 0; z < n_threads; z++) {//release GPU memory
		gasal_gpu_mem_free(&gpu_storage_arr[z]);;
	}
	total_time.Stop();
	string algo = al_type;
	string start_type[2] = {"without_start", "with_start"};
	al_type += "_";
	al_type += start_type[start_pos];
	double av_misc_time = 0.0;
	for (int i = 0; i < n_threads; ++i){
		av_misc_time += (thread_misc_time[i]/n_threads);
	}
	fprintf(stderr, "\n-------------------------------------------------------------------------------\n");
	fprintf(stderr, "%d threads performed %d %s alignments on GPU in %.3f milliseconds\n", n_threads, total_seqs, al_type.c_str(), total_time.GetTime() - av_misc_time);
	fprintf(stderr, "Total execution time (in milliseconds) including the time for filling the batches and printing (if any): %.3f\n", total_time.GetTime());
}
