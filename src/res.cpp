#include "gasal.h"

#include "args_parser.h"

#include "res.h"


gasal_res_t *gasal_res_new(uint32_t max_n_alns, Parameters *params, bool device = false)
{
	cudaError_t err;
	gasal_res_t *res = (gasal_res_t *)calloc(1, sizeof(gasal_res_t));

    CHECKCUDAERROR(cudaMalloc(&(res->aln_score), max_n_alns * sizeof(int32_t)));
    if (params->algo == GLOBAL) {
        res->query_batch_start = NULL;
        res->target_batch_start = NULL;
        res->query_batch_end = NULL;
        res->target_batch_end = NULL;
    /*
    // Deprecated. For semi-global you now need to know the start and stop positions.
    } else if (params->algo == SEMI_GLOBAL) {
        res->host_query_batch_start = NULL;
        res->host_query_batch_end = NULL;
        res->query_batch_start = NULL;
        res->query_batch_end = NULL;

        if (params->start_pos == WITH_START) {
            CHECKCUDAERROR(cudaMallocHost(&(res->host_target_batch_start),max_n_alns * sizeof(uint32_t)));
            CHECKCUDAERROR(cudaMallocHost(&(res->host_target_batch_end),max_n_alns * sizeof(uint32_t)));

            CHECKCUDAERROR(cudaMalloc(&(res->target_batch_start),max_n_alns * sizeof(uint32_t)));
            CHECKCUDAERROR(
                    cudaMalloc(&(res->target_batch_end),max_n_alns * sizeof(uint32_t)));
        } else {
            CHECKCUDAERROR(cudaMallocHost(&(res->host_target_batch_end),max_n_alns * sizeof(uint32_t)));
            CHECKCUDAERROR(cudaMalloc(&(res->target_batch_end),max_n_alns * sizeof(uint32_t)));
            res->host_target_batch_start = NULL;
            res->target_batch_start = NULL;
        }
    */
    } else {
        if (params->start_pos == WITH_START) {
            if (device)
            {
            CHECKCUDAERROR(cudaMalloc(&(res->query_batch_start),max_n_alns * sizeof(uint32_t)));
            CHECKCUDAERROR(cudaMalloc(&(res->target_batch_start),max_n_alns * sizeof(uint32_t)));
            CHECKCUDAERROR(cudaMalloc(&(res->query_batch_end),max_n_alns * sizeof(uint32_t)));
            CHECKCUDAERROR(cudaMalloc(&(res->target_batch_end),max_n_alns * sizeof(uint32_t)));
            } else {
            CHECKCUDAERROR(cudaMallocHost(&(res->query_batch_start),max_n_alns * sizeof(uint32_t)));
            CHECKCUDAERROR(cudaMallocHost(&(res->target_batch_start),max_n_alns * sizeof(uint32_t)));
            CHECKCUDAERROR(cudaMallocHost(&(res->query_batch_end),max_n_alns * sizeof(uint32_t)));
            CHECKCUDAERROR(cudaMallocHost(&(res->target_batch_end),max_n_alns * sizeof(uint32_t)));
            }

        } else {
            if (device)
            {
            CHECKCUDAERROR(cudaMalloc(&(res->query_batch_end),max_n_alns * sizeof(uint32_t)));
            CHECKCUDAERROR(cudaMalloc(&(res->target_batch_end),max_n_alns * sizeof(uint32_t)));
            } else {
            CHECKCUDAERROR(cudaMallocHost(&(res->query_batch_end),max_n_alns * sizeof(uint32_t)));
            CHECKCUDAERROR(cudaMallocHost(&(res->target_batch_end),max_n_alns * sizeof(uint32_t)));
            }

            res->query_batch_start = NULL;
            res->target_batch_start = NULL;
        }
        res->isDevice = device;
    }

	return res;
}


void gasal_res_destroy(gasal_res_t *res) 
{
    if (res == NULL)
        exit(1);
    
    if (res->isDevice)
    {
		if (res->aln_score != NULL) CHECKCUDAERROR(cudaFree(res->aln_score));
		if (res->query_batch_start != NULL) CHECKCUDAERROR(cudaFree(res->query_batch_start));
		if (res->target_batch_start != NULL) CHECKCUDAERROR(cudaFree(res->target_batch_start));
		if (res->query_batch_end != NULL) CHECKCUDAERROR(cudaFree(res->query_batch_end));
		if (res->target_batch_end != NULL) CHECKCUDAERROR(cudaFree(res->target_batch_end));
    } else {
		if (res->aln_score != NULL) CHECKCUDAERROR(cudaFreeHost(res->aln_score));
		if (res->query_batch_start != NULL) CHECKCUDAERROR(cudaFreeHost(res->query_batch_start));
		if (res->target_batch_start != NULL) CHECKCUDAERROR(cudaFreeHost(res->target_batch_start));
		if (res->query_batch_end != NULL) CHECKCUDAERROR(cudaFreeHost(res->query_batch_end));
		if (res->target_batch_end != NULL) CHECKCUDAERROR(cudaFreeHost(res->target_batch_end));
    }
    free(res);
}