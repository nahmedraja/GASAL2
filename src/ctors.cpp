
#include "gasal.h"

#include "args_parser.h"

#include "host_batch.h"

#include "res.h"

#include "ctors.h"

#include "interfaces.h"

#include <cmath>


gasal_gpu_storage_v gasal_init_gpu_storage_v(int n_streams) {
	gasal_gpu_storage_v v;
	v.a = (gasal_gpu_storage_t*)calloc(n_streams, sizeof(gasal_gpu_storage_t));
	v.n = n_streams;
	return v;

}


void gasal_init_streams(gasal_gpu_storage_v *gpu_storage_vec,  int max_query_len, int max_target_len, int max_n_alns,  Parameters *params) {

	cudaError_t err;
	int i;
	int max_query_len_8 = max_query_len % 8 ? max_query_len + (8 - (max_query_len % 8)) : max_query_len;
	int max_target_len_8 = max_target_len % 8 ? max_target_len + (8 - (max_target_len % 8)) : max_target_len;

	int host_max_query_batch_bytes = max_n_alns * max_query_len_8;
	int gpu_max_query_batch_bytes = max_n_alns * max_query_len_8;
	int host_max_target_batch_bytes =  max_n_alns * max_target_len_8;
	int gpu_max_target_batch_bytes =  max_n_alns * max_target_len_8;
	int host_max_n_alns = max_n_alns;
	int gpu_max_n_alns = max_n_alns;



	for (i = 0; i < gpu_storage_vec->n; i++) {

		gpu_storage_vec->a[i].extensible_host_unpacked_query_batch = gasal_host_batch_new(host_max_query_batch_bytes, 0);
		gpu_storage_vec->a[i].extensible_host_unpacked_target_batch = gasal_host_batch_new(host_max_target_batch_bytes, 0);

		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].unpacked_query_batch), gpu_max_query_batch_bytes * sizeof(uint8_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].unpacked_target_batch), gpu_max_target_batch_bytes * sizeof(uint8_t)));


		CHECKCUDAERROR(cudaHostAlloc(&(gpu_storage_vec->a[i].host_query_op), host_max_n_alns * sizeof(uint8_t), cudaHostAllocDefault));
		CHECKCUDAERROR(cudaHostAlloc(&(gpu_storage_vec->a[i].host_target_op), host_max_n_alns * sizeof(uint8_t), cudaHostAllocDefault));
		uint8_t *no_ops = NULL;
		no_ops = (uint8_t*) calloc(host_max_n_alns * sizeof(uint8_t), sizeof(uint8_t));
		gasal_op_fill(&(gpu_storage_vec->a[i]), no_ops, host_max_n_alns, QUERY);
		gasal_op_fill(&(gpu_storage_vec->a[i]), no_ops, host_max_n_alns, TARGET);
		free(no_ops);

		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].query_op), gpu_max_n_alns * sizeof(uint8_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].target_op), gpu_max_n_alns * sizeof(uint8_t)));



		if (params->isPacked)
		{
			gpu_storage_vec->a[i].packed_query_batch = (uint32_t *) gpu_storage_vec->a[i].unpacked_query_batch;
			gpu_storage_vec->a[i].packed_target_batch = (uint32_t *) gpu_storage_vec->a[i].unpacked_target_batch;

		} else {
			CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].packed_query_batch), (gpu_max_query_batch_bytes/8) * sizeof(uint32_t)));
			CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].packed_target_batch), (gpu_max_target_batch_bytes/8) * sizeof(uint32_t)));
		}

		if (params->algo == KSW)
		{
			CHECKCUDAERROR(cudaHostAlloc(&(gpu_storage_vec->a[i].host_seed_scores), host_max_n_alns * sizeof(uint32_t), cudaHostAllocDefault));
			CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].seed_scores), host_max_n_alns * sizeof(uint32_t)));
		} else {
			gpu_storage_vec->a[i].host_seed_scores = NULL;
			gpu_storage_vec->a[i].seed_scores = NULL;
		}


		CHECKCUDAERROR(cudaHostAlloc(&(gpu_storage_vec->a[i].host_query_batch_lens), host_max_n_alns * sizeof(uint32_t), cudaHostAllocDefault));
		CHECKCUDAERROR(cudaHostAlloc(&(gpu_storage_vec->a[i].host_target_batch_lens), host_max_n_alns * sizeof(uint32_t), cudaHostAllocDefault));
		CHECKCUDAERROR(cudaHostAlloc(&(gpu_storage_vec->a[i].host_query_batch_offsets), host_max_n_alns * sizeof(uint32_t), cudaHostAllocDefault));
		CHECKCUDAERROR(cudaHostAlloc(&(gpu_storage_vec->a[i].host_target_batch_offsets), host_max_n_alns * sizeof(uint32_t), cudaHostAllocDefault));

		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].query_batch_lens), gpu_max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].target_batch_lens), gpu_max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].query_batch_offsets), gpu_max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].target_batch_offsets), gpu_max_n_alns * sizeof(uint32_t)));
		

		gpu_storage_vec->a[i].host_res = gasal_res_new_host(host_max_n_alns, params);
		if(params->start_pos == WITH_TB) CHECKCUDAERROR(cudaHostAlloc(&(gpu_storage_vec->a[i].host_res->cigar), gpu_max_query_batch_bytes * sizeof(uint8_t),cudaHostAllocDefault));
		gpu_storage_vec->a[i].device_cpy = gasal_res_new_device_cpy(max_n_alns,  params);
		gpu_storage_vec->a[i].device_res = gasal_res_new_device(gpu_storage_vec->a[i].device_cpy);

		if (params->secondBest)
		{	
			gpu_storage_vec->a[i].host_res_second = gasal_res_new_host(host_max_n_alns,  params);
			gpu_storage_vec->a[i].device_cpy_second = gasal_res_new_device_cpy(host_max_n_alns, params);
			gpu_storage_vec->a[i].device_res_second = gasal_res_new_device(gpu_storage_vec->a[i].device_cpy_second);

		} else {
			gpu_storage_vec->a[i].host_res_second = NULL;
			gpu_storage_vec->a[i].device_cpy_second = NULL;
			gpu_storage_vec->a[i].device_res_second = NULL;
		}

		if (params->start_pos == WITH_TB) {
			gpu_storage_vec->a[i].packed_tb_matrix_size = ((uint32_t)ceil(((double)((uint64_t)max_query_len_8*(uint64_t)max_target_len_8))/32)) * gpu_max_n_alns;
			CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].packed_tb_matrices), gpu_storage_vec->a[i].packed_tb_matrix_size * sizeof(uint4)));
		}


		CHECKCUDAERROR(cudaStreamCreate(&(gpu_storage_vec->a[i].str)));
		gpu_storage_vec->a[i].is_free = 1;
		gpu_storage_vec->a[i].host_max_query_batch_bytes = host_max_query_batch_bytes;
		gpu_storage_vec->a[i].host_max_target_batch_bytes = host_max_target_batch_bytes;
		gpu_storage_vec->a[i].host_max_n_alns = host_max_n_alns;
		gpu_storage_vec->a[i].gpu_max_query_batch_bytes = gpu_max_query_batch_bytes;
		gpu_storage_vec->a[i].gpu_max_target_batch_bytes = gpu_max_target_batch_bytes;
		gpu_storage_vec->a[i].gpu_max_n_alns = gpu_max_n_alns;
		gpu_storage_vec->a[i].current_n_alns = 0;
	}
}

void gasal_destroy_streams(gasal_gpu_storage_v *gpu_storage_vec, Parameters *params) {

	cudaError_t err;

	int i;
	for (i = 0; i < gpu_storage_vec->n; i ++) {
		
		gasal_host_batch_destroy(gpu_storage_vec->a[i].extensible_host_unpacked_query_batch);
		gasal_host_batch_destroy(gpu_storage_vec->a[i].extensible_host_unpacked_target_batch);

		gasal_res_destroy_host(gpu_storage_vec->a[i].host_res);
		gasal_res_destroy_device(gpu_storage_vec->a[i].device_res, gpu_storage_vec->a[i].device_cpy);

		if (params->secondBest)
		{
			gasal_res_destroy_host(gpu_storage_vec->a[i].host_res_second);
			gasal_res_destroy_device(gpu_storage_vec->a[i].device_res_second, gpu_storage_vec->a[i].device_cpy_second);
		}


		if (!(params->algo == KSW))
		{
			if (gpu_storage_vec->a[i].seed_scores != NULL) CHECKCUDAERROR(cudaFree(gpu_storage_vec->a[i].seed_scores));
			if (gpu_storage_vec->a[i].host_seed_scores != NULL) CHECKCUDAERROR(cudaFreeHost(gpu_storage_vec->a[i].host_seed_scores));
		}

		if (gpu_storage_vec->a[i].query_op != NULL) CHECKCUDAERROR(cudaFree(gpu_storage_vec->a[i].query_op));
		if (gpu_storage_vec->a[i].target_op != NULL) CHECKCUDAERROR(cudaFree(gpu_storage_vec->a[i].target_op));
		if (gpu_storage_vec->a[i].host_query_op != NULL) CHECKCUDAERROR(cudaFreeHost(gpu_storage_vec->a[i].host_query_op));
		if (gpu_storage_vec->a[i].host_target_op != NULL) CHECKCUDAERROR(cudaFreeHost(gpu_storage_vec->a[i].host_target_op));

		if (gpu_storage_vec->a[i].host_query_batch_offsets != NULL) CHECKCUDAERROR(cudaFreeHost(gpu_storage_vec->a[i].host_query_batch_offsets));
		if (gpu_storage_vec->a[i].host_target_batch_offsets != NULL) CHECKCUDAERROR(cudaFreeHost(gpu_storage_vec->a[i].host_target_batch_offsets));
		if (gpu_storage_vec->a[i].host_query_batch_lens != NULL) CHECKCUDAERROR(cudaFreeHost(gpu_storage_vec->a[i].host_query_batch_lens));
		if (gpu_storage_vec->a[i].host_target_batch_lens != NULL) CHECKCUDAERROR(cudaFreeHost(gpu_storage_vec->a[i].host_target_batch_lens));
		if (gpu_storage_vec->a[i].host_res->cigar != NULL) CHECKCUDAERROR(cudaFreeHost(gpu_storage_vec->a[i].host_res->cigar));




		if (gpu_storage_vec->a[i].unpacked_query_batch != NULL) CHECKCUDAERROR(cudaFree(gpu_storage_vec->a[i].unpacked_query_batch));
		if (gpu_storage_vec->a[i].unpacked_target_batch != NULL) CHECKCUDAERROR(cudaFree(gpu_storage_vec->a[i].unpacked_target_batch));
		if (!(params->isPacked))
		{
			if (gpu_storage_vec->a[i].packed_query_batch != NULL) CHECKCUDAERROR(cudaFree(gpu_storage_vec->a[i].packed_query_batch));
			if (gpu_storage_vec->a[i].packed_target_batch != NULL) CHECKCUDAERROR(cudaFree(gpu_storage_vec->a[i].packed_target_batch));
		}


		if (gpu_storage_vec->a[i].query_batch_offsets != NULL) CHECKCUDAERROR(cudaFree(gpu_storage_vec->a[i].query_batch_offsets));
		if (gpu_storage_vec->a[i].target_batch_offsets != NULL) CHECKCUDAERROR(cudaFree(gpu_storage_vec->a[i].target_batch_offsets));
		if (gpu_storage_vec->a[i].query_batch_lens != NULL) CHECKCUDAERROR(cudaFree(gpu_storage_vec->a[i].query_batch_lens));
		if (gpu_storage_vec->a[i].target_batch_lens != NULL) CHECKCUDAERROR(cudaFree(gpu_storage_vec->a[i].target_batch_lens));
		if (gpu_storage_vec->a[i].packed_tb_matrices != NULL) CHECKCUDAERROR(cudaFree(gpu_storage_vec->a[i].packed_tb_matrices));

		if (gpu_storage_vec->a[i].str != NULL)CHECKCUDAERROR(cudaStreamDestroy(gpu_storage_vec->a[i].str));
	}



}


void gasal_destroy_gpu_storage_v(gasal_gpu_storage_v *gpu_storage_vec) {

	if(gpu_storage_vec->a != NULL) free(gpu_storage_vec->a);
}




// Deprecated
void gasal_gpu_mem_alloc(gasal_gpu_storage_t *gpu_storage, int gpu_max_query_batch_bytes, int gpu_max_target_batch_bytes, int gpu_max_n_alns, Parameters *params) {

	cudaError_t err;
	//	if (gpu_storage->gpu_max_query_batch_bytes % 8) {
	//		fprintf(stderr, "[GASAL ERROR:] max_query_batch_bytes=%d is not a multiple of 8\n", gpu_storage->gpu_max_query_batch_bytes % 8);
	//		exit(EXIT_FAILURE);
	//	}
	//	if (gpu_storage->gpu_max_target_batch_bytes % 8) {
	//		fprintf(stderr, "[GASAL ERROR:] max_target_batch_bytes=%d is not a multiple of 8\n", gpu_storage->gpu_max_target_batch_bytes % 8);
	//		exit(EXIT_FAILURE);
	//	}

	CHECKCUDAERROR(cudaMalloc(&(gpu_storage->unpacked_query_batch), gpu_max_query_batch_bytes * sizeof(uint8_t)));
	CHECKCUDAERROR(cudaMalloc(&(gpu_storage->unpacked_target_batch), gpu_max_target_batch_bytes * sizeof(uint8_t)));

	CHECKCUDAERROR(cudaMalloc(&(gpu_storage->packed_query_batch), (gpu_max_query_batch_bytes/8) * sizeof(uint32_t)));
	CHECKCUDAERROR(cudaMalloc(&(gpu_storage->packed_target_batch), (gpu_max_target_batch_bytes/8) * sizeof(uint32_t)));

	CHECKCUDAERROR(cudaMalloc(&(gpu_storage->query_batch_lens), gpu_max_n_alns * sizeof(uint32_t)));
	CHECKCUDAERROR(cudaMalloc(&(gpu_storage->target_batch_lens), gpu_max_n_alns * sizeof(uint32_t)));
	CHECKCUDAERROR(cudaMalloc(&(gpu_storage->query_batch_offsets), gpu_max_n_alns * sizeof(uint32_t)));
	CHECKCUDAERROR(cudaMalloc(&(gpu_storage->target_batch_offsets), gpu_max_n_alns * sizeof(uint32_t)));

	gpu_storage->device_res = gasal_res_new_device(gpu_storage->device_cpy);

	gpu_storage->gpu_max_query_batch_bytes = gpu_max_query_batch_bytes;
	gpu_storage->gpu_max_target_batch_bytes = gpu_max_target_batch_bytes;
	gpu_storage->gpu_max_n_alns = gpu_max_n_alns;

}

// Deprecated
void gasal_gpu_mem_free(gasal_gpu_storage_t *gpu_storage, Parameters *params) {

	cudaError_t err;

	if (gpu_storage->unpacked_query_batch != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->unpacked_query_batch));
	if (gpu_storage->unpacked_target_batch != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->unpacked_target_batch));
	if (gpu_storage->packed_query_batch != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->packed_query_batch));
	if (gpu_storage->packed_target_batch != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->packed_target_batch));
	if (gpu_storage->query_batch_offsets != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->query_batch_offsets));
	if (gpu_storage->target_batch_offsets != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->target_batch_offsets));
	if (gpu_storage->query_batch_lens != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->query_batch_lens));
	if (gpu_storage->target_batch_lens != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->target_batch_lens));
	
	gasal_res_destroy_device(gpu_storage->device_res,gpu_storage->device_cpy);
	if (params->secondBest)
	{
		gasal_res_destroy_device(gpu_storage->device_res_second, gpu_storage->device_cpy_second);
	}
}
