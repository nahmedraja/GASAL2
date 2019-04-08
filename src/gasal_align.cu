#include "gasal.h"
#include "args_parser.h"
#include "res.h"
#include "gasal_align.h"
#include "gasal_kernels.h"



inline void gasal_kernel_launcher(int32_t N_BLOCKS, int32_t BLOCKDIM, algo_type algo, comp_start start, gasal_gpu_storage_t *gpu_storage, int32_t actual_n_alns, int32_t k_band, data_source semiglobal_skipping_head, data_source semiglobal_skipping_tail, Bool secondBest) 
{
	switch(algo)
	{
		
		KERNEL_SWITCH(LOCAL,		start, semiglobal_skipping_head, semiglobal_skipping_tail, secondBest);
		KERNEL_SWITCH(SEMI_GLOBAL,  start, semiglobal_skipping_head, semiglobal_skipping_tail, secondBest);		// MACRO that expands all 32 semi-global kernels
		KERNEL_SWITCH(GLOBAL,		start, semiglobal_skipping_head, semiglobal_skipping_tail, secondBest);
		KERNEL_SWITCH(KSW,			start, semiglobal_skipping_head, semiglobal_skipping_tail, secondBest);
		KERNEL_SWITCH(BANDED,		start, semiglobal_skipping_head, semiglobal_skipping_tail, secondBest);
		default:
		break;

	}

}


//GASAL2 asynchronous (a.k.a non-blocking) alignment function
void gasal_aln_async(gasal_gpu_storage_t *gpu_storage, const uint32_t actual_query_batch_bytes, const uint32_t actual_target_batch_bytes, const uint32_t actual_n_alns, Parameters *params) {

	cudaError_t err;
	if (actual_n_alns <= 0) {
		fprintf(stderr, "[GASAL ERROR:] actual_n_alns <= 0\n");
		exit(EXIT_FAILURE);
	}
	if (actual_query_batch_bytes <= 0) {
		fprintf(stderr, "[GASAL ERROR:] actual_query_batch_bytes <= 0\n");
		exit(EXIT_FAILURE);
	}
	if (actual_target_batch_bytes <= 0) {
		fprintf(stderr, "[GASAL ERROR:] actual_target_batch_bytes <= 0\n");
		exit(EXIT_FAILURE);
	}

	if (actual_query_batch_bytes % 8) {
		fprintf(stderr, "[GASAL ERROR:] actual_query_batch_bytes=%d is not a multiple of 8\n", actual_query_batch_bytes);
		exit(EXIT_FAILURE);
	}
	if (actual_target_batch_bytes % 8) {
		fprintf(stderr, "[GASAL ERROR:] actual_target_batch_bytes=%d is not a multiple of 8\n", actual_target_batch_bytes);
		exit(EXIT_FAILURE);
	}

	if (actual_query_batch_bytes > gpu_storage->host_max_query_batch_bytes) {
				fprintf(stderr, "[GASAL ERROR:] actual_query_batch_bytes(%d) > host_max_query_batch_bytes(%d)\n", actual_query_batch_bytes, gpu_storage->host_max_query_batch_bytes);
				exit(EXIT_FAILURE);
	}

	if (actual_target_batch_bytes > gpu_storage->host_max_target_batch_bytes) {
			fprintf(stderr, "[GASAL ERROR:] actual_target_batch_bytes(%d) > host_max_target_batch_bytes(%d)\n", actual_target_batch_bytes, gpu_storage->host_max_target_batch_bytes);
			exit(EXIT_FAILURE);
	}

	if (actual_n_alns > gpu_storage->host_max_n_alns) {
			fprintf(stderr, "[GASAL ERROR:] actual_n_alns(%d) > host_max_n_alns(%d)\n", actual_n_alns, gpu_storage->host_max_n_alns);
			exit(EXIT_FAILURE);
	}

	//--------------if pre-allocated memory is less, allocate more--------------------------
	if (gpu_storage->gpu_max_query_batch_bytes < actual_query_batch_bytes) {

		int i = 2;
		while ( (gpu_storage->gpu_max_query_batch_bytes * i) < actual_query_batch_bytes) i++;

		fprintf(stderr, "[GASAL WARNING:] actual_query_batch_bytes(%d) > Allocated GPU memory (gpu_max_query_batch_bytes=%d). Therefore, allocating %d bytes on GPU (gpu_max_query_batch_bytes=%d). Performance may be lost if this is repeated many times.\n", actual_query_batch_bytes, gpu_storage->gpu_max_query_batch_bytes, gpu_storage->gpu_max_query_batch_bytes*i, gpu_storage->gpu_max_query_batch_bytes*i);

		gpu_storage->gpu_max_query_batch_bytes = gpu_storage->gpu_max_query_batch_bytes * i;

		if (gpu_storage->unpacked_query_batch != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->unpacked_query_batch));
		if (gpu_storage->packed_query_batch != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->packed_query_batch));

		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->unpacked_query_batch), gpu_storage->gpu_max_query_batch_bytes * sizeof(uint8_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->packed_query_batch), (gpu_storage->gpu_max_query_batch_bytes/8) * sizeof(uint32_t)));

	}

	if (gpu_storage->gpu_max_target_batch_bytes < actual_target_batch_bytes) {

		int i = 2;
		while ( (gpu_storage->gpu_max_target_batch_bytes * i) < actual_target_batch_bytes) i++;
		
		fprintf(stderr, "[GASAL WARNING:] actual_target_batch_bytes(%d) > Allocated GPU memory (gpu_max_target_batch_bytes=%d). Therefore, allocating %d bytes on GPU (gpu_max_target_batch_bytes=%d). Performance may be lost if this is repeated many times.\n", actual_target_batch_bytes, gpu_storage->gpu_max_target_batch_bytes, gpu_storage->gpu_max_target_batch_bytes*i, gpu_storage->gpu_max_target_batch_bytes*i);

		gpu_storage->gpu_max_target_batch_bytes = gpu_storage->gpu_max_target_batch_bytes * i;

		if (gpu_storage->unpacked_target_batch != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->unpacked_target_batch));
		if (gpu_storage->packed_target_batch != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->packed_target_batch));

		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->unpacked_target_batch), gpu_storage->gpu_max_target_batch_bytes * sizeof(uint8_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->packed_target_batch), (gpu_storage->gpu_max_target_batch_bytes/8) * sizeof(uint32_t)));


	}

	if (gpu_storage->gpu_max_n_alns < actual_n_alns) {

		int i = 2;
		while ( (gpu_storage->gpu_max_n_alns * i) < actual_n_alns) i++;
		
		fprintf(stderr, "[GASAL WARNING:] actual_n_alns(%d) > gpu_max_n_alns(%d). Therefore, allocating memory for %d alignments on  GPU (gpu_max_n_alns=%d). Performance may be lost if this is repeated many times.\n", actual_n_alns, gpu_storage->gpu_max_n_alns, gpu_storage->gpu_max_n_alns*i, gpu_storage->gpu_max_n_alns*i);

		gpu_storage->gpu_max_n_alns = gpu_storage->gpu_max_n_alns * i;

		if (gpu_storage->query_batch_offsets != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->query_batch_offsets));
		if (gpu_storage->target_batch_offsets != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->target_batch_offsets));
		if (gpu_storage->query_batch_lens != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->query_batch_lens));
		if (gpu_storage->target_batch_lens != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->target_batch_lens));

		if (gpu_storage->seed_scores != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->seed_scores));

		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->query_batch_lens), gpu_storage->gpu_max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->target_batch_lens), gpu_storage->gpu_max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->query_batch_offsets), gpu_storage->gpu_max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->target_batch_offsets), gpu_storage->gpu_max_n_alns * sizeof(uint32_t)));

		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->seed_scores), gpu_storage->gpu_max_n_alns * sizeof(uint32_t)));

		gasal_res_destroy_device(gpu_storage->device_res, gpu_storage->device_cpy);
		gpu_storage->device_cpy = gasal_res_new_device_cpy(gpu_storage->gpu_max_n_alns, params);
		gpu_storage->device_res = gasal_res_new_device(gpu_storage->device_cpy);

		if (params->secondBest)
		{
			gasal_res_destroy_device(gpu_storage->device_res_second, gpu_storage->device_cpy_second);
			gpu_storage->device_cpy_second = gasal_res_new_device_cpy(gpu_storage->gpu_max_n_alns, params);
			gpu_storage->device_res_second = gasal_res_new_device(gpu_storage->device_cpy_second);
		}

	}
	//------------------------------------------

	//------------------------launch copying of sequence batches from CPU to GPU---------------------------

	// here you can track the evolution of your data structure processing with the printer: gasal_host_batch_printall(current);

	host_batch_t *current = gpu_storage->extensible_host_unpacked_query_batch;
	while (current != NULL)
	{
		if (current->next != NULL ) 
		{
			CHECKCUDAERROR(cudaMemcpyAsync( &(gpu_storage->unpacked_query_batch[current->offset]), 
											current->data, 
											current->next->offset - current->offset,
											cudaMemcpyHostToDevice, 
											gpu_storage->str ) );
			
		} else {
			// it's the last page to copy
			CHECKCUDAERROR(cudaMemcpyAsync( &(gpu_storage->unpacked_query_batch[current->offset]), 
											current->data, 
											actual_query_batch_bytes - current->offset, 
											cudaMemcpyHostToDevice, 
											gpu_storage->str ) );
		}
		current = current->next;
	}

	current = gpu_storage->extensible_host_unpacked_target_batch;
	while (current != NULL)
	{
		if (current->next != NULL ) {
			CHECKCUDAERROR(cudaMemcpyAsync( &(gpu_storage->unpacked_target_batch[current->offset]), 
											current->data, 
											current->next->offset - current->offset,
											cudaMemcpyHostToDevice, 
											gpu_storage->str ) );

		} else {
			// it's the last page to copy
			CHECKCUDAERROR(cudaMemcpyAsync( &(gpu_storage->unpacked_target_batch[current->offset]), 
											current->data, 
											actual_target_batch_bytes - current->offset, 
											cudaMemcpyHostToDevice, 
											gpu_storage->str ) );
		}
		current = current->next;
	}

	//-----------------------------------------------------------------------------------------------------------
	// TODO: Adjust the block size depending on the kernel execution.
	
    uint32_t BLOCKDIM = 128;
    uint32_t N_BLOCKS = (actual_n_alns + BLOCKDIM - 1) / BLOCKDIM;

    int query_batch_tasks_per_thread = (int)ceil((double)actual_query_batch_bytes/(8*BLOCKDIM*N_BLOCKS));
    int target_batch_tasks_per_thread = (int)ceil((double)actual_target_batch_bytes/(8*BLOCKDIM*N_BLOCKS));


    //-------------------------------------------launch packing kernel


	if (!(params->isPacked))
	{
		gasal_pack_kernel<<<N_BLOCKS, BLOCKDIM, 0, gpu_storage->str>>>((uint32_t*)(gpu_storage->unpacked_query_batch),
		(uint32_t*)(gpu_storage->unpacked_target_batch), gpu_storage->packed_query_batch, gpu_storage->packed_target_batch,
		query_batch_tasks_per_thread, target_batch_tasks_per_thread, actual_query_batch_bytes/4, actual_target_batch_bytes/4);
		cudaError_t pack_kernel_err = cudaGetLastError();
		if ( cudaSuccess != pack_kernel_err )
		{
		fprintf(stderr, "[GASAL CUDA ERROR:] %s(CUDA error no.=%d). Line no. %d in file %s\n", cudaGetErrorString(pack_kernel_err), pack_kernel_err,  __LINE__, __FILE__);
		exit(EXIT_FAILURE);
		}
	}
    

	// We could reverse-complement before packing, but we would get 2x more read-writes to memory.

    //----------------------launch copying of sequence offsets and lengths from CPU to GPU--------------------------------------
    CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->query_batch_lens, gpu_storage->host_query_batch_lens, actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice, gpu_storage->str));
    CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->target_batch_lens, gpu_storage->host_target_batch_lens, actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice,  gpu_storage->str));
    CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->query_batch_offsets, gpu_storage->host_query_batch_offsets, actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice,  gpu_storage->str));
	CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->target_batch_offsets, gpu_storage->host_target_batch_offsets, actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice,  gpu_storage->str));
	
	// if needed copy seed scores
	if (params->algo == KSW)
	{
		if (gpu_storage->seed_scores == NULL)
		{
			fprintf(stderr, "seed_scores == NULL\n");
			
		}
		if (gpu_storage->host_seed_scores == NULL)
		{
			fprintf(stderr, "host_seed_scores == NULL\n");
		}
		if (gpu_storage->seed_scores == NULL || gpu_storage->host_seed_scores == NULL)
			exit(EXIT_FAILURE);

		CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->seed_scores, gpu_storage->host_seed_scores, actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice, gpu_storage->str));
	}
    //--------------------------------------------------------------------------------------------------------------------------

	//----------------------launch copying of sequence operations (reverse/complement) from CPU to GPU--------------------------
	if (params->isReverseComplement)
	{
		CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->query_op, gpu_storage->host_query_op, actual_n_alns * sizeof(uint8_t), cudaMemcpyHostToDevice,  gpu_storage->str));
		CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->target_op, gpu_storage->host_target_op, actual_n_alns * sizeof(uint8_t), cudaMemcpyHostToDevice,  gpu_storage->str));	
		//--------------------------------------launch reverse-complement kernel------------------------------------------------------
		gasal_reversecomplement_kernel<<<N_BLOCKS, BLOCKDIM, 0, gpu_storage->str>>>(gpu_storage->packed_query_batch, gpu_storage->packed_target_batch, gpu_storage->query_batch_lens,
			gpu_storage->target_batch_lens, gpu_storage->query_batch_offsets, gpu_storage->target_batch_offsets, gpu_storage->query_op, gpu_storage->target_op, actual_n_alns);
		cudaError_t reversecomplement_kernel_err = cudaGetLastError();
		if ( cudaSuccess != reversecomplement_kernel_err )
		{
			 fprintf(stderr, "[GASAL CUDA ERROR:] %s(CUDA error no.=%d). Line no. %d in file %s\n", cudaGetErrorString(reversecomplement_kernel_err), reversecomplement_kernel_err,  __LINE__, __FILE__);
			 exit(EXIT_FAILURE);
		}
	
	}
	
    //--------------------------------------launch alignment kernels--------------------------------------------------------------
	gasal_kernel_launcher(N_BLOCKS, BLOCKDIM, params->algo, params->start_pos, gpu_storage, actual_n_alns, params->k_band, params->semiglobal_skipping_head, params->semiglobal_skipping_tail, params->secondBest);


        //-----------------------------------------------------------------------------------------------------------------------
    cudaError_t aln_kernel_err = cudaGetLastError();
    if ( cudaSuccess != aln_kernel_err )
    {
    	fprintf(stderr, "[GASAL CUDA ERROR:] %s(CUDA error no.=%d). Line no. %d in file %s\n", cudaGetErrorString(aln_kernel_err), aln_kernel_err,  __LINE__, __FILE__);
    	exit(EXIT_FAILURE);
    }

    //------------------------0launch the copying of alignment results from GPU to CPU--------------------------------------
    if (gpu_storage->host_res->aln_score != NULL && gpu_storage->device_cpy->aln_score != NULL) 
		CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->host_res->aln_score, gpu_storage->device_cpy->aln_score, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage->str));
    
	if (gpu_storage->host_res->query_batch_start != NULL && gpu_storage->device_cpy->query_batch_start != NULL) 
		CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->host_res->query_batch_start, gpu_storage->device_cpy->query_batch_start, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage->str));
    
	if (gpu_storage->host_res->target_batch_start != NULL && gpu_storage->device_cpy->target_batch_start != NULL) 
		CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->host_res->target_batch_start, gpu_storage->device_cpy->target_batch_start, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage->str));
    
	if (gpu_storage->host_res->query_batch_end != NULL && gpu_storage->device_cpy->query_batch_end != NULL) 
		CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->host_res->query_batch_end, gpu_storage->device_cpy->query_batch_end, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage->str));
    
	if (gpu_storage->host_res->target_batch_end != NULL && gpu_storage->device_cpy->target_batch_end != NULL) 
		CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->host_res->target_batch_end, gpu_storage->device_cpy->target_batch_end, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage->str));
	//-----------------------------------------------------------------------------------------------------------------------
	

	// not really needed to filter with params->secondBest, since all the pointers will be null and non-initialized.
	if (params->secondBest)
	{	
		if (gpu_storage->host_res_second->aln_score != NULL && gpu_storage->device_cpy_second->aln_score != NULL) 
			CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->host_res_second->aln_score, gpu_storage->device_cpy_second->aln_score, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage->str));
    
		if (gpu_storage->host_res_second->query_batch_start != NULL && gpu_storage->device_cpy_second->query_batch_start != NULL) 
			CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->host_res_second->query_batch_start, gpu_storage->device_cpy_second->query_batch_start, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage->str));
		
		if (gpu_storage->host_res_second->target_batch_start != NULL && gpu_storage->device_cpy_second->target_batch_start != NULL) 
			CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->host_res_second->target_batch_start, gpu_storage->device_cpy_second->target_batch_start, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage->str));
		
		if (gpu_storage->host_res_second->query_batch_end != NULL && gpu_storage->device_cpy_second->query_batch_end != NULL) 
			CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->host_res_second->query_batch_end, gpu_storage->device_cpy_second->query_batch_end, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage->str));
		
		if (gpu_storage->host_res_second->target_batch_end != NULL && gpu_storage->device_cpy_second->target_batch_end != NULL) 
			CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->host_res_second->target_batch_end, gpu_storage->device_cpy_second->target_batch_end, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage->str));
	}

    gpu_storage->is_free = 0; //set the availability of current stream to false

}


int gasal_is_aln_async_done(gasal_gpu_storage_t *gpu_storage) 
{
	cudaError_t err;
	if(gpu_storage->is_free == 1) return -2;//if no work is launced in this stream, return -2
	err = cudaStreamQuery(gpu_storage->str);//check to see if the stream is finished
	if (err != cudaSuccess ) {
		if (err == cudaErrorNotReady) return -1;
		else{
			fprintf(stderr, "[GASAL CUDA ERROR:] %s(CUDA error no.=%d). Line no. %d in file %s\n", cudaGetErrorString(err), err,  __LINE__, __FILE__);
			exit(EXIT_FAILURE);
		}
	}
	gpu_storage->is_free = 1;

	return 0;
}


void gasal_copy_subst_scores(gasal_subst_scores *subst){

	cudaError_t err;
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaGapO, &(subst->gap_open), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaGapExtend, &(subst->gap_extend), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	int32_t gapoe = (subst->gap_open + subst->gap_extend);
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaGapOE, &(gapoe), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaMatchScore, &(subst->match), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaMismatchScore, &(subst->mismatch), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	return;
}

