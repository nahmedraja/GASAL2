#ifndef __GASAL_ALIGN_H__
#define __GASAL_ALIGN_H__

// SEMI_GLOBAL Kernels generation - read from the bottom one, all the way up. (the most specialized ones are written before the ones that call them)

#define SEMIGLOBAL_KERNEL_CALL(a,s,h,t) \
	case t:\
		gasal_semi_global_kernel<Int2Type<a>, Int2Type<s>, Int2Type<h>, Int2Type<t>><<<N_BLOCKS, BLOCKDIM, 0, gpu_storage->str>>>(gpu_storage->packed_query_batch, gpu_storage->packed_target_batch, gpu_storage->query_batch_lens, gpu_storage->target_batch_lens, gpu_storage->query_batch_offsets, gpu_storage->target_batch_offsets, gpu_storage->aln_score, gpu_storage->query_batch_end, gpu_storage->target_batch_end, gpu_storage->query_batch_start, gpu_storage->target_batch_start, actual_n_alns); \
	break;

#define SWITCH_SEMI_GLOBAL_TAIL(a,s,h,t) \
	case h:\
	switch(t) { \
		SEMIGLOBAL_KERNEL_CALL(a,s,h,NONE)\
		SEMIGLOBAL_KERNEL_CALL(a,s,h,QUERY)\
		SEMIGLOBAL_KERNEL_CALL(a,s,h,TARGET)\
		SEMIGLOBAL_KERNEL_CALL(a,s,h,BOTH)\
	}\
	break;

#define SWITCH_SEMI_GLOBAL_HEAD(a,s,h,t) \
	case s:\
	switch(h) { \
		SWITCH_SEMI_GLOBAL_TAIL(a,s,NONE,t)\
		SWITCH_SEMI_GLOBAL_TAIL(a,s,QUERY,t)\
		SWITCH_SEMI_GLOBAL_TAIL(a,s,TARGET,t)\
		SWITCH_SEMI_GLOBAL_TAIL(a,s,BOTH,t)\
	} \
	break;\

// ALGORITHMS Kernels generation. Allows to have a single line written for all kernels calls. The switch-cases are MACRO-generated.

#define SWITCH_SEMI_GLOBAL(a,s,h,t) SWITCH_SEMI_GLOBAL_HEAD(a,s,h,t)

#define SWITCH_LOCAL(a,s,h,t) \
    case s:\
        gasal_local_kernel<Int2Type<LOCAL>, Int2Type<s>><<<N_BLOCKS, BLOCKDIM, 0, gpu_storage->str>>>(gpu_storage->packed_query_batch, gpu_storage->packed_target_batch, gpu_storage->query_batch_lens, gpu_storage->target_batch_lens, gpu_storage->query_batch_offsets, gpu_storage->target_batch_offsets, gpu_storage->aln_score, gpu_storage->query_batch_end, gpu_storage->target_batch_end, gpu_storage->query_batch_start, gpu_storage->target_batch_start, actual_n_alns); \
    break;

#define SWITCH_MICROLOCAL(a,s,h,t) \
    case s:\
        gasal_local_kernel<Int2Type<MICROLOCAL>, Int2Type<s>><<<N_BLOCKS, BLOCKDIM, 0, gpu_storage->str>>>(gpu_storage->packed_query_batch, gpu_storage->packed_target_batch, gpu_storage->query_batch_lens, gpu_storage->target_batch_lens, gpu_storage->query_batch_offsets, gpu_storage->target_batch_offsets, gpu_storage->aln_score, gpu_storage->query_batch_end, gpu_storage->target_batch_end, gpu_storage->query_batch_start, gpu_storage->target_batch_start, actual_n_alns);\
    break;

#define SWITCH_GLOBAL(a,s,h,t) \
    case s:\
        gasal_global_kernel<<<N_BLOCKS, BLOCKDIM, 0, gpu_storage->str>>>(gpu_storage->packed_query_batch, gpu_storage->packed_target_batch, gpu_storage->query_batch_lens,gpu_storage->target_batch_lens, gpu_storage->query_batch_offsets, gpu_storage->target_batch_offsets, gpu_storage->aln_score, actual_n_alns);\
    break;


#define SWITCH_BANDED(a,s,h,t) \
    case s:\
        gasal_banded_tiled_kernel<<<N_BLOCKS, BLOCKDIM, 0, gpu_storage->str>>>(gpu_storage->packed_query_batch, gpu_storage->packed_target_batch, gpu_storage->query_batch_lens,gpu_storage->target_batch_lens, gpu_storage->query_batch_offsets, gpu_storage->target_batch_offsets, gpu_storage->aln_score, gpu_storage->query_batch_end, gpu_storage->target_batch_end, actual_n_alns, k_band>>3); \
    break;\

// MACRO calls : general call (bottom, should be used), and first level WITH / WITHOUT_START switch call (top)

#define SWITCH_START(a,s,h,t) \
    switch(s){\
        SWITCH_## a(a,WITH_START,h,t)\
        SWITCH_## a(a,WITHOUT_START,h,t)\
    }

#define KERNEL_SWITCH(a,s,h,t) \
    case a:\
        SWITCH_START(a,s,h,t)\
    break;


/* // Deprecated
void gasal_aln(gasal_gpu_storage_t *gpu_storage, const uint8_t *query_batch, const uint32_t *query_batch_offsets, const uint32_t *query_batch_lens, const uint8_t *target_batch, const uint32_t *target_batch_offsets, const uint32_t *target_batch_lens,   const uint32_t actual_query_batch_bytes, const uint32_t actual_target_batch_bytes, const uint32_t actual_n_alns, int32_t *host_aln_score, int32_t *host_query_batch_start, int32_t *host_target_batch_start, int32_t *host_query_batch_end, int32_t *host_target_batch_end,  algo_type algo, comp_start start, int32_t k_band);
*/

void gasal_copy_subst_scores(gasal_subst_scores *subst);

void gasal_aln_async(gasal_gpu_storage_t *gpu_storage, const uint32_t actual_query_batch_bytes, const uint32_t actual_target_batch_bytes, const uint32_t actual_n_alns, Parameters *params);

inline void gasal_kernel_launcher(int32_t N_BLOCKS, int32_t BLOCKDIM, algo_type algo, comp_start start, gasal_gpu_storage_t *gpu_storage, int32_t actual_n_alns, int32_t k_band);

int gasal_is_aln_async_done(gasal_gpu_storage_t *gpu_storage);

#endif