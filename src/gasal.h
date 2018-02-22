#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "/usr/local/cuda-8.0/include/cuda_runtime.h"

#ifndef MAX_BATCH1_LEN
#define MAX_BATCH1_LEN 304
#endif

#ifndef MAX_BATCH2_LEN
#define MAX_BATCH2_LEN 600
#endif

#ifndef MAX_SEQ_LEN
#define MAX_SEQ_LEN (MAX_BATCH1_LEN > MAX_BATCH2_LEN ? MAX_BATCH1_LEN : MAX_BATCH2_LEN)
#endif

#ifndef HOST_MALLOC_SAFETY_FACTOR
#define HOST_MALLOC_SAFETY_FACTOR 5
#endif

//typedef int32_t gasal_error_t;


enum comp_start{
	WITH_START,
	WITHOUT_START
};

enum algo_type{
	LOCAL,
	GLOBAL,
	SEMI_GLOBAL
};



typedef struct {
	uint8_t *unpacked1;
	uint8_t *unpacked2;
	uint32_t *packed1_4bit;
	uint32_t *packed2_4bit;
	uint32_t *offsets1;
	uint32_t *offsets2;
	uint32_t *lens1;
	uint32_t *lens2;
	uint8_t *host_unpacked1;
	uint8_t *host_unpacked2;
	uint32_t *host_offsets1;
	uint32_t *host_offsets2;
	uint32_t *host_lens1;
	uint32_t *host_lens2;
	int32_t *aln_score;
	int32_t *batch1_end;
	int32_t *batch2_end;
	int32_t *batch1_start;
	int32_t *batch2_start;
	int32_t *results;
	int32_t *host_aln_score;
	int32_t *host_batch1_end;
	int32_t *host_batch2_end;
	int32_t *host_batch1_start;
	int32_t *host_batch2_start;
	int32_t *host_results;
	uint32_t max_batch1_bytes;
	uint32_t max_batch2_bytes;
	uint32_t host_max_batch1_bytes;
	uint32_t host_max_batch2_bytes;
	uint32_t max_n_alns;
	uint32_t host_max_n_alns;
	uint32_t n_alns;
	uint64_t max_gpu_malloc_size;
	cudaStream_t str;
	int is_free;
	int is_host_mem_alloc;
	int is_gpu_mem_alloc;
	int is_host_mem_alloc_contig;
	int is_gpu_mem_alloc_contig;

} gasal_gpu_storage_t;

typedef struct {
	int n;
	gasal_gpu_storage_t *a;
}gasal_gpu_storage_v;



typedef struct{
	int32_t match;
	int32_t mismatch;
	int32_t gap_open;
	int32_t gap_extend;
} gasal_subst_scores;



#ifdef __cplusplus
extern "C" {
#endif

/*void*/ gasal_gpu_storage_t* gasal_aln(const uint8_t *batch1, const uint32_t *batch1_offsets, const uint32_t *batch1_lens, const uint8_t *batch2, const uint32_t *batch2_offsets, const uint32_t *batch2_lens, const uint32_t batch1_bytes, const uint32_t batch2_bytes, const uint32_t n_alns, int32_t *host_aln_score, int32_t *host_batch1_start, int32_t *host_batch2_start, int32_t *host_batch1_end, int32_t *host_batch2_end, int algo, int start);

gasal_gpu_storage_t* gasal_aln_async(const uint8_t *batch1, const uint32_t *batch1_offsets, const uint32_t *batch1_lens, const uint8_t *batch2, const uint32_t *batch2_offsets, const uint32_t *batch2_lens,   const uint32_t batch1_bytes, const uint32_t batch2_bytes, const uint32_t n_alns, int32_t *host_aln_score, int32_t *host_batch1_start, int32_t *host_batch2_start, int32_t *host_batch1_end, int32_t *host_batch2_end,  int algo, int start);

gasal_gpu_storage_v gasal_init_gpu_storage_v(int n_streams);

void gasal_aln_async_new(gasal_gpu_storage_t *gpu_storage, const uint32_t actual_batch1_bytes, const uint32_t actual_batch2_bytes, const uint32_t actual_n_alns, int algo, int start);

void gasal_aln_async_new2(gasal_gpu_storage_t *gpu_storage, const uint32_t actual_batch1_bytes, const uint32_t actual_batch2_bytes, const uint32_t actual_n_alns, int algo, int start);

void gasal_aln_async_new3(gasal_gpu_storage_t *gpu_storage, const uint32_t actual_batch1_bytes, const uint32_t actual_batch2_bytes, const uint32_t actual_n_alns, int algo, int start);

void gasal_init_streams(gasal_gpu_storage_v  *gpu_storage_vec, int max_batch1_bytes, int max_batch2_bytes, int max_n_alns, int algo, int start);

void gasal_init_streams_new(gasal_gpu_storage_v *gpu_storage_vec, int max_batch1_bytes, int max_batch2_bytes, int max_n_alns, int algo, int start);

void gasal_init_streams_with_host_malloc(gasal_gpu_storage_v *gpu_storage_vec, int host_max_batch1_bytes,  int max_batch1_bytes,  int host_max_batch2_bytes, int max_batch2_bytes, int host_max_n_alns, int max_n_alns, int algo, int start);

void gasal_gpu_mem_alloc_contig (gasal_gpu_storage_t *gpu_storage, const uint32_t actual_batch1_bytes, const uint32_t actual_batch2_bytes, const uint32_t actual_n_alns, int algo, int start);

void gasal_destroy_streams(gasal_gpu_storage_v *gpu_storage_vec);

void gasal_destroy_streams_with_host_malloc(gasal_gpu_storage_v *gpu_storage_vec);

void gasal_destroy_gpu_storage_v(gasal_gpu_storage_v *gpu_storage_vec);

void gasal_aln_imp(const uint8_t *batch1, const uint32_t *batch1_offsets, const uint32_t *batch1_lens, const uint8_t *batch2, const uint32_t *batch2_offsets, const uint32_t *batch2_lens,   const uint32_t actual_batch1_bytes, const uint32_t actual_batch2_bytes, const uint32_t actual_n_alns, int32_t *host_aln_score, int32_t *host_batch1_start, int32_t *host_batch2_start, int32_t *host_batch1_end, int32_t *host_batch2_end,  int algo, int start, gasal_gpu_storage_t *gpu_storage);

int gasal_is_aln_async_done(gasal_gpu_storage_t *gpu_storage);

int gasal_is_aln_async_done_new(gasal_gpu_storage_t *gpu_storage);

int gasal_is_aln_async_done_new2(gasal_gpu_storage_t *gpu_storage);

void gasal_get_aln_async_results(int32_t *host_aln_score, int32_t *host_batch1_start, int32_t *host_batch2_start, int32_t *host_batch1_end, int32_t *host_batch2_end, gasal_gpu_storage_t *gpu_storage);

void gasal_free_gpu_storage(gasal_gpu_storage_t *gpu_storage);

void gasal_stream_destroy(gasal_gpu_storage_t *gpu_storage);

void gasal_gpu_mem_alloc(gasal_gpu_storage_t *gpu_storage, const uint32_t actual_batch1_bytes, const uint32_t actual_batch2_bytes, const uint32_t actual_n_alns, int algo, int start);

void gasal_host_mem_alloc(gasal_gpu_storage_t *gpu_storage, int max_batch1_bytes, int max_batch2_bytes, int max_n_alns, int algo, int start);

void gasal_host_mem_free(gasal_gpu_storage_t *gpu_storage);

void gasal_aln_imp_mem_alloc(gasal_gpu_storage_t *gpu_storage, int algo, int start);

void gasal_aln_imp_mem_free(gasal_gpu_storage_t *gpu_storage);

void gasal_copy_subst_scores(gasal_subst_scores *subst);

void gasal_gpu_mem_free(gasal_gpu_storage_v *gpu_storage_vec);

//void gasal_host_malloc(void **mem_ptr, uint32_t n_bytes);

//void gasal_host_free(void *mem_ptr);

void gasal_host_malloc_uint32(uint32_t **mem_ptr, uint32_t n_bytes);

void gasal_host_malloc_int32(int32_t **mem_ptr, uint32_t n_bytes);

void gasal_host_malloc_uint8(uint8_t **mem_ptr, uint32_t n_bytes);

void gasal_host_free_uint32(uint32_t *mem_free_ptr);

void gasal_host_free_int32(int32_t *mem_free_ptr);

void gasal_host_free_uint8(uint8_t *mem_free_ptr);

#ifdef __cplusplus
}
#endif

