
#include "gasal.h"

#include "args_parser.h"

#include "host_batch.h"

#include "res.h"

#include "ctors.h"




gasal_gpu_storage_v gasal_init_gpu_storage_v(int n_streams) {
	gasal_gpu_storage_v v;
	v.a = (gasal_gpu_storage_t*)calloc(n_streams, sizeof(gasal_gpu_storage_t));
	v.n = n_streams;
	return v;

}


void gasal_init_streams(gasal_gpu_storage_v *gpu_storage_vec, int host_max_query_batch_bytes,  int gpu_max_query_batch_bytes,  int host_max_target_batch_bytes, int gpu_max_target_batch_bytes, int host_max_n_alns, int gpu_max_n_alns, Parameters *params) {

	cudaError_t err;
	int i;
	for (i = 0; i < gpu_storage_vec->n; i++) {

		gpu_storage_vec->a[i].extensible_host_unpacked_query_batch = gasal_host_batch_new(host_max_query_batch_bytes, 0);
		gpu_storage_vec->a[i].extensible_host_unpacked_target_batch = gasal_host_batch_new(host_max_target_batch_bytes, 0);

		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].unpacked_query_batch), gpu_max_query_batch_bytes * sizeof(uint8_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].unpacked_target_batch), gpu_max_target_batch_bytes * sizeof(uint8_t)));


		CHECKCUDAERROR(cudaHostAlloc(&(gpu_storage_vec->a[i].host_query_op), host_max_n_alns * sizeof(uint8_t), cudaHostAllocDefault));
		CHECKCUDAERROR(cudaHostAlloc(&(gpu_storage_vec->a[i].host_target_op), host_max_n_alns * sizeof(uint8_t), cudaHostAllocDefault));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].query_op), gpu_max_n_alns * sizeof(uint8_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].target_op), gpu_max_n_alns * sizeof(uint8_t)));



		if (params->isPacked)
		{
			// assign 
			gpu_storage_vec->a[i].packed_query_batch = (uint32_t *) gpu_storage_vec->a[i].unpacked_query_batch;
			gpu_storage_vec->a[i].packed_target_batch = (uint32_t *) gpu_storage_vec->a[i].unpacked_target_batch;

		} else {
			CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].packed_query_batch), (gpu_max_query_batch_bytes/8) * sizeof(uint32_t)));
			CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].packed_target_batch), (gpu_max_target_batch_bytes/8) * sizeof(uint32_t)));
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
		gpu_storage_vec->a[i].device_cpy = gasal_res_new_device_cpy(host_max_n_alns, params);
		gpu_storage_vec->a[i].device_res = gasal_res_new_device(gpu_storage_vec->a[i].device_cpy);

		if (params->secondBest)
		{	
			gpu_storage_vec->a[i].host_res_second = gasal_res_new_host(host_max_n_alns, params);
			gpu_storage_vec->a[i].device_cpy_second = gasal_res_new_device_cpy(host_max_n_alns, params);
			gpu_storage_vec->a[i].device_res_second = gasal_res_new_device(gpu_storage_vec->a[i].device_cpy_second);

		} else {
			gpu_storage_vec->a[i].host_res_second = NULL;
			gpu_storage_vec->a[i].device_cpy_second = NULL;
			gpu_storage_vec->a[i].device_res_second = NULL;
		}


		CHECKCUDAERROR(cudaStreamCreate(&(gpu_storage_vec->a[i].str)));
		gpu_storage_vec->a[i].is_free = 1;
		gpu_storage_vec->a[i].host_max_query_batch_bytes = host_max_query_batch_bytes;
		gpu_storage_vec->a[i].host_max_target_batch_bytes = host_max_target_batch_bytes;
		gpu_storage_vec->a[i].host_max_n_alns = host_max_n_alns;
		gpu_storage_vec->a[i].gpu_max_query_batch_bytes = gpu_max_query_batch_bytes;
		gpu_storage_vec->a[i].gpu_max_target_batch_bytes = gpu_max_target_batch_bytes;
		gpu_storage_vec->a[i].gpu_max_n_alns = gpu_max_n_alns;

	}


}


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


void gasal_gpu_mem_free(gasal_gpu_storage_t *gpu_storage) {

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

		if (gpu_storage_vec->a[i].query_op != NULL) CHECKCUDAERROR(cudaFree(gpu_storage_vec->a[i].query_op));
		if (gpu_storage_vec->a[i].target_op != NULL) CHECKCUDAERROR(cudaFree(gpu_storage_vec->a[i].target_op));
		if (gpu_storage_vec->a[i].host_query_op != NULL) CHECKCUDAERROR(cudaFreeHost(gpu_storage_vec->a[i].host_query_op));
		if (gpu_storage_vec->a[i].host_target_op != NULL) CHECKCUDAERROR(cudaFreeHost(gpu_storage_vec->a[i].host_target_op));

		if (gpu_storage_vec->a[i].host_query_batch_offsets != NULL) CHECKCUDAERROR(cudaFreeHost(gpu_storage_vec->a[i].host_query_batch_offsets));
		if (gpu_storage_vec->a[i].host_target_batch_offsets != NULL) CHECKCUDAERROR(cudaFreeHost(gpu_storage_vec->a[i].host_target_batch_offsets));
		if (gpu_storage_vec->a[i].host_query_batch_lens != NULL) CHECKCUDAERROR(cudaFreeHost(gpu_storage_vec->a[i].host_query_batch_lens));
		if (gpu_storage_vec->a[i].host_target_batch_lens != NULL) CHECKCUDAERROR(cudaFreeHost(gpu_storage_vec->a[i].host_target_batch_lens));




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

		if (gpu_storage_vec->a[i].str != NULL)CHECKCUDAERROR(cudaStreamDestroy(gpu_storage_vec->a[i].str));
	}



}


void gasal_destroy_gpu_storage_v(gasal_gpu_storage_v *gpu_storage_vec) {

	if(gpu_storage_vec->a != NULL) free(gpu_storage_vec->a);
}

