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

#define NB_STREAMS 2

#define DEBUG

#define MAX(a,b) (a>b ? a : b)




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
	uint32_t maximum_sequence_length = 0;
	uint32_t target_seqs_len = 0;
	uint32_t query_seqs_len = 0;
	cerr << "Loading files...." << endl;

	/*
		Reads FASTA files and fill the corresponding buffers.
		FASTA files contain sequences that are usually on separate lines.
		The file reader detects a '>' then concatenates all the following lines into one sequence, until the next '>' or EOF.
		See more about FASTA format : https://en.wikipedia.org/wiki/FASTA_format
	*/
	
	int seq_begin=0;
	while (getline(query_batch_fasta, query_batch_line) && getline(target_batch_fasta, target_batch_line)) { 

		//load sequences from the files
		if (query_batch_line[0] == '>' && target_batch_line[0] == '>') {
			query_headers.push_back(query_batch_line.substr(1));
			target_headers.push_back(target_batch_line.substr(1));

			if (seq_begin == 2) {
				// a sequence was already being read. Now it's done, so we should find its length.
				target_seqs_len += (target_seqs.back()).length();
				query_seqs_len += (query_seqs.back()).length();
				maximum_sequence_length = MAX((target_seqs.back()).length(), maximum_sequence_length);
				maximum_sequence_length = MAX((query_seqs.back()).length(), maximum_sequence_length);
			}
			seq_begin = 1;
			total_seqs++;
		} else if (seq_begin == 1) {
			query_seqs.push_back(query_batch_line);
			target_seqs.push_back(target_batch_line);
			seq_begin=2;
		} else if (seq_begin == 2) {
			query_seqs.back() += query_batch_line;
			target_seqs.back() += target_batch_line;
		} else { // should never happen but always put an else, for safety...
			seq_begin = 0;
			cerr << "Batch1 and target_batch files should be fasta having same number of sequences" << endl;
			exit(EXIT_FAILURE);
		}
	}


	// Check maximum sequence length one more time, to check the last read sequence:
	target_seqs_len += (target_seqs.back()).length();
	query_seqs_len += (query_seqs.back()).length();
	maximum_sequence_length = MAX((target_seqs.back()).length(), maximum_sequence_length);
	maximum_sequence_length = MAX((query_seqs.back()).length(), maximum_sequence_length);

	#ifdef DEBUG
		fprintf(stderr, "Size of read batches are: query=%d, target=%d. maximum_sequence_length=%d\n", query_seqs_len, target_seqs_len, maximum_sequence_length);
	 #endif
	
	// here you should know all your data size.

	int *thread_seqs_idx = (int*)malloc(n_threads*sizeof(int));
	int *thread_n_seqs = (int*)malloc(n_threads*sizeof(int));
	int *thread_n_batchs = (int*)malloc(n_threads*sizeof(int));
	double *thread_misc_time = (double*)calloc(n_threads, sizeof(double));

	int thread_batch_size = (int)ceil((double)total_seqs/n_threads);
	int n_seqs_alloc = 0;
	for (int i = 0; i < n_threads; i++){//distribute the sequences among the threads equally
		thread_seqs_idx[i] = n_seqs_alloc;
		if (n_seqs_alloc + thread_batch_size < total_seqs) thread_n_seqs[i] = thread_batch_size;
		else thread_n_seqs[i] = total_seqs - n_seqs_alloc;
		thread_n_batchs[i] = (int)ceil((double)thread_n_seqs[i]/(target_seqs.size() / NB_STREAMS));
		n_seqs_alloc += thread_n_seqs[i];

	}

	cerr << "Processing..." << endl;

	Timer total_time;
	total_time.Start();
	omp_set_num_threads(n_threads);
	gasal_gpu_storage_v *gpu_storage_vecs =  (gasal_gpu_storage_v*)calloc(n_threads, sizeof(gasal_gpu_storage_v));
	for (int z = 0; z < n_threads; z++) {
		gpu_storage_vecs[z] = gasal_init_gpu_storage_v(NB_STREAMS);// creating NB_STREAMS streams per thread

		/* 
			About memory sizes:
			The required memory is the total size of the batch + its padding, divided by the number of streams. 
			The worst case would be that every sequence has to be padded with 7 'N', since they must have a length multiple of 8.
			Even though the memory can be dynamically expanded both for Host and Device, it is advised to start with a memory large enough so that these expansions rarely occur (for better performance.)
			Modifying the factor '1' in front of each size lets you see how GASAL2 expands the memory when needed.
		*/

		//initializing the streams by allocating the required CPU and GPU memory
		// note: the calculations of the detailed sizes to allocate could be done on the library side (to hide it from the user's perspective)
		gasal_init_streams(&(gpu_storage_vecs[z]), 
							0.4 * (query_seqs_len +7*total_seqs) / (NB_STREAMS) , 
							1 * (query_seqs_len +7*total_seqs)  / (NB_STREAMS) , 
							1 * (target_seqs_len +7*total_seqs)/ (NB_STREAMS),
							1 * (target_seqs_len +7*total_seqs) / (NB_STREAMS) , 
							(target_seqs.size() / NB_STREAMS), // maximum number of alignments is bigger on target than on query side.
							(target_seqs.size() / NB_STREAMS), 
							LOCAL, 
							WITH_START);

	}
	#ifdef DEBUG
		fprintf(stderr, "size of host_unpack_query is %d\n", (query_seqs_len +7*total_seqs) / (NB_STREAMS) );
	#endif

	#pragma omp parallel
	{
	int n_seqs = thread_n_seqs[omp_get_thread_num()];//number of sequences allocated to this thread
	int curr_idx = thread_seqs_idx[omp_get_thread_num()];//number of sequences allocated to this thread
	int seqs_done = 0;
	int n_batchs_done = 0;

	struct gpu_batch{ //a struct to hold data structures of a stream
			gasal_gpu_storage_t *gpu_storage; //the struct that holds the GASAL2 data structures
			int n_seqs_batch;//number of sequences in the batch (<= (target_seqs.size() / NB_STREAMS))
			int batch_start;//starting index of batch
	};

	#ifdef DEBUG
		fprintf(stderr, "Number of gpu_batch in gpu_batch_arr : %d\n", gpu_storage_vecs[omp_get_thread_num()].n);
		fprintf(stderr, "Number of gpu_storage_vecs in a gpu_batch : %d\n", omp_get_thread_num()+1);
	#endif

	gpu_batch gpu_batch_arr[gpu_storage_vecs[omp_get_thread_num()].n];

	for(int z = 0; z < gpu_storage_vecs[omp_get_thread_num()].n; z++) {
		gpu_batch_arr[z].gpu_storage = &(gpu_storage_vecs[omp_get_thread_num()].a[z]);

	}

	if (n_seqs > 0) {
		while (n_batchs_done < thread_n_batchs[omp_get_thread_num()]) {
			int gpu_batch_arr_idx = 0;
			//------------checking the availability of a "free" stream"-----------------
			while(gpu_batch_arr_idx < gpu_storage_vecs[omp_get_thread_num()].n && (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->is_free != 1) {
				gpu_batch_arr_idx++;
			}
			//---------------------------------------------------------------------------
			// Needs to re-allocate the linked list in case the stream is recycled.
			gasal_host_batch_recycle((gpu_batch_arr[gpu_batch_arr_idx].gpu_storage));

			if (seqs_done < n_seqs && gpu_batch_arr_idx < gpu_storage_vecs[omp_get_thread_num()].n) {
					uint32_t query_batch_idx = 0;
					uint32_t target_batch_idx = 0;
					unsigned int j = 0;
					//-----------Create a batch of sequences to be aligned on the GPU. The batch contains (target_seqs.size() / NB_STREAMS) number of sequences-----------------------
					for (int i = curr_idx; seqs_done < n_seqs && j < (target_seqs.size() / NB_STREAMS); i++, j++, seqs_done++) {

						(gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_query_batch_offsets[j] = query_batch_idx;
						(gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_target_batch_offsets[j] = target_batch_idx;

						/*
							All the filling is moved on the library size, to take care of the memory size and expansions (when needed).
							The function gasal_host_batch_fill takes care of how to fill, how much to pad with 'N', and how to deal with memory. 
							It's the same function for query and target, and you only need to set the final flag to either ; this avoides code duplication.
							The way the host memory is filled changes the current _idx (it's increased by size, and by the padding). That's why it's returned by the function.
						*/

						query_batch_idx = gasal_host_batch_fill(gpu_batch_arr[gpu_batch_arr_idx].gpu_storage, 
										query_batch_idx, 
										query_seqs[i].c_str(), 
										query_seqs[i].size(),
										QUERY);

						target_batch_idx = gasal_host_batch_fill(gpu_batch_arr[gpu_batch_arr_idx].gpu_storage, 
										target_batch_idx, 
										target_seqs[i].c_str(), 
										target_seqs[i].size(),
										TARGET);

						(gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_query_batch_lens[j] = query_seqs[i].size();
						(gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_target_batch_lens[j] = target_seqs[i].size();

					}
					gpu_batch_arr[gpu_batch_arr_idx].n_seqs_batch = j;
					uint32_t query_batch_bytes = query_batch_idx;
					uint32_t target_batch_bytes = target_batch_idx;
					gpu_batch_arr[gpu_batch_arr_idx].batch_start = curr_idx;
					curr_idx += (target_seqs.size() / NB_STREAMS);

					//----------------------------------------------------------------------------------------------------
					//-----------------calling the GASAL2 non-blocking alignment function---------------------------------
					if (al_type.compare("local") == 0){
						if(start_pos){
							gasal_aln_async(gpu_batch_arr[gpu_batch_arr_idx].gpu_storage, query_batch_bytes, target_batch_bytes, gpu_batch_arr[gpu_batch_arr_idx].n_seqs_batch,  LOCAL, WITH_START);
						}
						else {
							gasal_aln_async(gpu_batch_arr[gpu_batch_arr_idx].gpu_storage, query_batch_bytes, target_batch_bytes, gpu_batch_arr[gpu_batch_arr_idx].n_seqs_batch, LOCAL, WITHOUT_START);
						}
					}
					else if (al_type.compare("semi_global") == 0) {
						if (start_pos) {
							gasal_aln_async(gpu_batch_arr[gpu_batch_arr_idx].gpu_storage, query_batch_bytes, target_batch_bytes, gpu_batch_arr[gpu_batch_arr_idx].n_seqs_batch, SEMI_GLOBAL, WITH_START);

						} else {
							gasal_aln_async(gpu_batch_arr[gpu_batch_arr_idx].gpu_storage, query_batch_bytes, target_batch_bytes, gpu_batch_arr[gpu_batch_arr_idx].n_seqs_batch, SEMI_GLOBAL, WITHOUT_START);

						}
					} else {
						gasal_aln_async(gpu_batch_arr[gpu_batch_arr_idx].gpu_storage, query_batch_bytes, target_batch_bytes, gpu_batch_arr[gpu_batch_arr_idx].n_seqs_batch, GLOBAL, WITHOUT_START);

					}
					//---------------------------------------------------------------------------------
			}

			//-------------------------------print alignment results----------------------------------------
			if(print_out) {
				gpu_batch_arr_idx = 0;
				while (gpu_batch_arr_idx < gpu_storage_vecs[omp_get_thread_num()].n) {//loop through all the streams and print the results
																					  //of the finished streams.
					if (gasal_is_aln_async_done(gpu_batch_arr[gpu_batch_arr_idx].gpu_storage) == 0) {
						int j = 0;

#pragma omp critical
						for (int i = gpu_batch_arr[gpu_batch_arr_idx].batch_start; j < gpu_batch_arr[gpu_batch_arr_idx].n_seqs_batch; i++, j++) {
							if(al_type.compare("local") == 0) {
								if (start_pos){
									fprintf(stdout, "query_name=%s\ttarget_name=%s\tscore=%d\tquery_batch_start=%d\ttarget_batch_start=%d\tquery_batch_end=%d\ttarget_batch_end=%d\n", query_headers[i].c_str(), target_headers[i].c_str(),(gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_aln_score[j], (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_query_batch_start[j],
											(gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_target_batch_start[j], (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_query_batch_end[j], (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_target_batch_end[j]);
								}
								else {
									fprintf(stdout, "query_name=%s\ttarget_name=%s\tscore=%d\tquery_batch_end=%d\ttarget_batch_end=%d\n", query_headers[i].c_str(), target_headers[i].c_str(), (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_aln_score[j], (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_query_batch_end[j], (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_target_batch_end[j]);
								}
							} else if(al_type.compare("semi_global") == 0) {
								if (start_pos){
									fprintf(stdout, "query_name=%s\ttarget_name=%s\tscore=%d\ttarget_batch_start=%d\ttarget_batch_end=%d\n", query_headers[i].c_str(), target_headers[i].c_str(), (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_aln_score[j], (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_target_batch_start[j], (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_target_batch_end[j]);

								}
								else {
									fprintf(stdout, "query_name=%s\ttarget_name=%s\tscore=%d\ttarget_batch_end=%d\n", query_headers[i].c_str(), target_headers[i].c_str(), (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_aln_score[j], (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_target_batch_end[j]);
								}
							}   else{
								fprintf(stdout, "query_name=%s\ttarget_name=%s\tscore=%d\n", query_headers[i].c_str(), target_headers[i].c_str(), (gpu_batch_arr[gpu_batch_arr_idx].gpu_storage)->host_aln_score[j]);
							}
						}
						n_batchs_done++;
					}
					gpu_batch_arr_idx++;
				}
			}
			//----------------------------------------------------------------------------------------------------

		}
	}


	}
	for (int z = 0; z < n_threads; z++) {
		gasal_destroy_streams(&(gpu_storage_vecs[z]));
		gasal_destroy_gpu_storage_v(&(gpu_storage_vecs[z]));
	}
	free(gpu_storage_vecs);
	total_time.Stop();
	string algo = al_type;
	string start_type[2] = {"without_start", "with_start"};
	al_type += "_";
	al_type += start_type[start_pos];
	double av_misc_time = 0.0;
	for (int i = 0; i < n_threads; ++i){
		av_misc_time += (thread_misc_time[i]/n_threads);
	}
	fprintf(stderr, "\nDone\n");
	fprintf(stderr, "Total execution time (in milliseconds): %.3f\n", total_time.GetTime());
}
