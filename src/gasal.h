#ifndef __GASAL_H__
#define __GASAL_H__

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "/usr/local/cuda-9.2/targets/x86_64-linux/include/cuda_runtime.h"


#ifndef HOST_MALLOC_SAFETY_FACTOR
#define HOST_MALLOC_SAFETY_FACTOR 5
#endif

#define CHECKCUDAERROR(error) \
		do{\
			err = error;\
			if (cudaSuccess != err ) { \
				fprintf(stderr, "[GASAL CUDA ERROR:] %s(CUDA error no.=%d). Line no. %d in file %s\n", cudaGetErrorString(err), err,  __LINE__, __FILE__); \
				exit(EXIT_FAILURE);\
			}\
		}while(0)\


inline int CudaCheckKernelLaunch()
{
	cudaError err = cudaGetLastError();
	if ( cudaSuccess != err )
	{
		return -1;

	}

	return 0;
}


enum comp_start{
	WITH_START,
	WITHOUT_START
};

enum data_source{
	NONE,
	QUERY,
	TARGET,
	BOTH
};

enum algo_type{
	UNKNOWN,
	GLOBAL,
	SEMI_GLOBAL,
	LOCAL,
	BANDED,
	MICROLOCAL,
	FIXEDBAND
};

enum operation_on_seq{
	FORWARD_NATURAL,
	REVERSE_NATURAL,
	FORWARD_COMPLEMENT,
	REVERSE_COMPLEMENT,
};

// data structure of linked list to allow extension of memory on host side
struct host_batch{
	uint8_t *data;
	uint32_t offset;
	struct host_batch* next;
};
typedef struct host_batch host_batch_t;

//stream data
typedef struct {
	uint8_t *unpacked_query_batch;
	uint8_t *unpacked_target_batch;
	uint32_t *packed_query_batch;
	uint32_t *packed_target_batch;
	uint32_t *query_batch_offsets;
	uint32_t *target_batch_offsets;
	uint32_t *query_batch_lens;
	uint32_t *target_batch_lens;
	
	host_batch_t *extensible_host_unpacked_query_batch;
	host_batch_t *extensible_host_unpacked_target_batch;

	uint8_t *host_query_op;
	uint8_t *host_target_op;
	uint8_t *query_op;
	uint8_t *target_op;

	uint32_t *host_query_batch_offsets;
	uint32_t *host_target_batch_offsets;
	uint32_t *host_query_batch_lens;
	uint32_t *host_target_batch_lens;
	int32_t *aln_score;
	int32_t *query_batch_end;
	int32_t *target_batch_end;
	int32_t *query_batch_start;
	int32_t *target_batch_start;
	int32_t *host_aln_score;
	int32_t *host_query_batch_end;
	int32_t *host_target_batch_end;
	int32_t *host_query_batch_start;
	int32_t *host_target_batch_start;
	uint32_t gpu_max_query_batch_bytes;
	uint32_t gpu_max_target_batch_bytes;

	uint32_t host_max_query_batch_bytes;
	uint32_t host_max_target_batch_bytes;
	
	uint32_t gpu_max_n_alns;
	uint32_t host_max_n_alns;
	cudaStream_t str;
	int is_free;

} gasal_gpu_storage_t;

//vector of streams
typedef struct {
	int n;
	gasal_gpu_storage_t *a;
}gasal_gpu_storage_v;


//match/mismatch and gap penalties
typedef struct{
	int32_t match;
	int32_t mismatch;
	int32_t gap_open;
	int32_t gap_extend;
} gasal_subst_scores;




#endif
