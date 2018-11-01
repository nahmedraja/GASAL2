#ifndef __CTORS_H__
#define __CTORS_H__


gasal_gpu_storage_v gasal_init_gpu_storage_v(int n_streams);

void gasal_init_streams(gasal_gpu_storage_v *gpu_storage_vec, int host_max_query_batch_bytes,  int gpu_max_query_batch_bytes,  int host_max_target_batch_bytes, int gpu_max_target_batch_bytes, int host_max_n_alns, int gpu_max_n_alns, Parameters *params);

void gasal_gpu_mem_alloc(gasal_gpu_storage_t *gpu_storage, int gpu_max_query_batch_bytes, int gpu_max_target_batch_bytes, int gpu_max_n_alns, algo_type algo, comp_start start);

void gasal_gpu_mem_free(gasal_gpu_storage_t *gpu_storage);

void gasal_destroy_streams(gasal_gpu_storage_v *gpu_storage_vec, Parameters *params);

void gasal_destroy_gpu_storage_v(gasal_gpu_storage_v *gpu_storage_vec);

#endif