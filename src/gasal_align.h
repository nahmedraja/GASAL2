#ifndef __GASAL_ALIGN_H__
#define __GASAL_ALIGN_H__

// SEMI_GLOBAL Kernels generation - read from the lowest one, all the way up. (the most specialized ones must be written before the ones that call them)

#define SEMIGLOBAL_KERNEL_CALL(a,s,h,t) \
	case t:\
		gasal_semi_global_kernel<Int2Type<a>, Int2Type<s>, Int2Type<h>, Int2Type<t>><<<N_BLOCKS, BLOCKDIM, 0, gpu_storage->str>>>(gpu_storage->packed_query_batch, gpu_storage->packed_target_batch, gpu_storage->query_batch_lens, gpu_storage->target_batch_lens, gpu_storage->query_batch_offsets, gpu_storage->target_batch_offsets, gpu_storage->aln_score, gpu_storage->query_batch_end, gpu_storage->target_batch_end, gpu_storage->query_batch_start, gpu_storage->target_batch_start, actual_n_alns); \
	break;

#define SEMIGLOBAL_SWITCH_TAIL(a,s,h,t) \
	case h:\
	switch(t) { \
		SEMIGLOBAL_KERNEL_CALL(a,s,h,NONE)\
		SEMIGLOBAL_KERNEL_CALL(a,s,h,QUERY)\
		SEMIGLOBAL_KERNEL_CALL(a,s,h,TARGET)\
		SEMIGLOBAL_KERNEL_CALL(a,s,h,BOTH)\
	}\
	break;

#define SEMIGLOBAL_SWITCH_HEAD(a,s,h,t) \
	case s:\
	switch(h) { \
		SEMIGLOBAL_SWITCH_TAIL(a,s,NONE,t)\
		SEMIGLOBAL_SWITCH_TAIL(a,s,QUERY,t)\
		SEMIGLOBAL_SWITCH_TAIL(a,s,TARGET,t)\
		SEMIGLOBAL_SWITCH_TAIL(a,s,BOTH,t)\
	} \
	break;\

#define SEMIGLOBAL_SWITCH(a,s,h,t) \
	switch(s) { \
		SEMIGLOBAL_SWITCH_HEAD(a,WITH_START,h,t)\
		SEMIGLOBAL_SWITCH_HEAD(a,WITHOUT_START,h,t)\
	}


/* // Deprecated
void gasal_aln(gasal_gpu_storage_t *gpu_storage, const uint8_t *query_batch, const uint32_t *query_batch_offsets, const uint32_t *query_batch_lens, const uint8_t *target_batch, const uint32_t *target_batch_offsets, const uint32_t *target_batch_lens,   const uint32_t actual_query_batch_bytes, const uint32_t actual_target_batch_bytes, const uint32_t actual_n_alns, int32_t *host_aln_score, int32_t *host_query_batch_start, int32_t *host_target_batch_start, int32_t *host_query_batch_end, int32_t *host_target_batch_end,  algo_type algo, comp_start start, int32_t k_band);
*/

void gasal_copy_subst_scores(gasal_subst_scores *subst);

void gasal_aln_async(gasal_gpu_storage_t *gpu_storage, const uint32_t actual_query_batch_bytes, const uint32_t actual_target_batch_bytes, const uint32_t actual_n_alns, algo_type algo, comp_start start, const int32_t k_band, const data_source semiglobal_skipping_head, const data_source semiglobal_skipping_tail);

inline void gasal_kernel_launcher(int32_t N_BLOCKS, int32_t BLOCKDIM, algo_type algo, comp_start start, gasal_gpu_storage_t *gpu_storage, int32_t actual_n_alns, int32_t k_band);

int gasal_is_aln_async_done(gasal_gpu_storage_t *gpu_storage);

#endif