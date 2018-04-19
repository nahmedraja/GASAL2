#include "gasal.h"




#define CHECKCUDAERROR(error) \
		do{\
			err = error;\
			if (cudaSuccess != err ) { \
				fprintf(stderr, "Cuda error:%d(%s) at line no. %d in file %s\n", err, cudaGetErrorString(err), __LINE__, __FILE__); \
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




#include "gasal_kernels_inl.h"


gasal_gpu_storage_v gasal_init_gpu_storage_v(int n_streams) {
	gasal_gpu_storage_v v;
	v.a = (gasal_gpu_storage_t*)calloc(n_streams, sizeof(gasal_gpu_storage_t));
	v.n = n_streams;
	return v;

}



//GASAL2 blocking alignment function
void gasal_aln(gasal_gpu_storage_t *gpu_storage, const uint8_t *batch1, const uint32_t *batch1_offsets, const uint32_t *batch1_lens, const uint8_t *batch2, const uint32_t *batch2_offsets, const uint32_t *batch2_lens, const uint32_t actual_batch1_bytes, const uint32_t actual_batch2_bytes, const uint32_t actual_n_alns, int32_t *host_aln_score, int32_t *host_batch1_start, int32_t *host_batch2_start, int32_t *host_batch1_end, int32_t *host_batch2_end,  int algo, int start) {

	cudaError_t err;
	if (actual_n_alns <= 0) {
		fprintf(stderr, "Must perform at least 1 alignment (n_alns > 0)\n");
		exit(EXIT_FAILURE);
	}
	if (actual_batch1_bytes <= 0) {
		fprintf(stderr, "Number of batch1_bytes should be greater than 0\n");
		exit(EXIT_FAILURE);
	}
	if (actual_batch2_bytes <= 0) {
		fprintf(stderr, "Number of batch2_bytes should be greater than 0\n");
		exit(EXIT_FAILURE);
	}

	if (actual_batch1_bytes % 8) {
		fprintf(stderr, "Number of batch1_bytes should be multiple of 8\n");
		exit(EXIT_FAILURE);
	}
	if (actual_batch2_bytes % 8) {
		fprintf(stderr, "Number of batch2_bytes should be multiple of 8\n");
		exit(EXIT_FAILURE);

	}
	//--------------if pre-allocated memory is less, allocate more--------------------------
	if (gpu_storage->gpu_max_batch1_bytes < actual_batch1_bytes) {
		fprintf(stderr, "max_batch1_bytes(%d) should be >= acutal_batch1_bytes(%d) \n", gpu_storage->gpu_max_batch1_bytes, actual_batch1_bytes);

		int i = 2;
		while ( (gpu_storage->gpu_max_batch1_bytes * i) < actual_batch1_bytes) i++;

		fprintf(stderr, "Therefore mallocing with max_batch1_bytes=%d \n", gpu_storage->gpu_max_batch1_bytes*i);
		gpu_storage->gpu_max_batch1_bytes = gpu_storage->gpu_max_batch1_bytes * i;

		if (gpu_storage->unpacked1 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->unpacked1));
		if (gpu_storage->packed1_4bit != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->packed1_4bit));

		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->unpacked1), gpu_storage->gpu_max_batch1_bytes * sizeof(uint8_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->packed1_4bit), (gpu_storage->gpu_max_batch1_bytes/8) * sizeof(uint32_t)));




	}

	if (gpu_storage->gpu_max_batch2_bytes < actual_batch2_bytes) {
		fprintf(stderr, "max_batch2_bytes(%d) should be >= acutal_batch2_bytes(%d) \n", gpu_storage->gpu_max_batch2_bytes, actual_batch2_bytes);

		int i = 2;
		while ( (gpu_storage->gpu_max_batch2_bytes * i) < actual_batch2_bytes) i++;

		fprintf(stderr, "Therefore mallocing with max_batch2_bytes=%d \n", gpu_storage->gpu_max_batch2_bytes*i);
		gpu_storage->gpu_max_batch2_bytes = gpu_storage->gpu_max_batch2_bytes * i;

		if (gpu_storage->unpacked2 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->unpacked2));
		if (gpu_storage->packed2_4bit != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->packed2_4bit));

		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->unpacked2), gpu_storage->gpu_max_batch2_bytes * sizeof(uint8_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->packed2_4bit), (gpu_storage->gpu_max_batch2_bytes/8) * sizeof(uint32_t)));


	}

	if (gpu_storage->gpu_max_n_alns < actual_n_alns) {
		fprintf(stderr, "Maximum possible number of alignment tasks(%d) should be >= acutal number of alignment tasks(%d) \n", gpu_storage->gpu_max_n_alns, actual_n_alns);

		int i = 2;
		while ( (gpu_storage->gpu_max_n_alns * i) < actual_n_alns) i++;

		fprintf(stderr, "Therefore mallocing with max_n_alns=%d \n", gpu_storage->gpu_max_n_alns*i);
		gpu_storage->gpu_max_n_alns = gpu_storage->gpu_max_n_alns * i;

		if (gpu_storage->offsets1 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->offsets1));
		if (gpu_storage->offsets2 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->offsets2));
		if (gpu_storage->lens1 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->lens1));
		if (gpu_storage->lens2 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->lens2));
		if (gpu_storage->aln_score != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->aln_score));
		if (gpu_storage->batch1_start != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->batch1_start));
		if (gpu_storage->batch2_start != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->batch2_start));
		if (gpu_storage->batch1_end != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->batch1_end));
		if (gpu_storage->batch2_end != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->batch2_end));

		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->lens1), gpu_storage->gpu_max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->lens2), gpu_storage->gpu_max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->offsets1), gpu_storage->gpu_max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->offsets2), gpu_storage->gpu_max_n_alns * sizeof(uint32_t)));

		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->aln_score),gpu_storage->gpu_max_n_alns * sizeof(int32_t)));
		if (algo == GLOBAL) {
			gpu_storage->batch1_start = NULL;
			gpu_storage->batch1_end = NULL;
			gpu_storage->batch2_start = NULL;
			gpu_storage->batch2_end = NULL;
		} else {
			CHECKCUDAERROR(
					cudaMalloc(&(gpu_storage->batch2_end),
							gpu_storage->gpu_max_n_alns * sizeof(uint32_t)));
			if (start == WITH_START) {
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage->batch2_start),
								gpu_storage->gpu_max_n_alns * sizeof(uint32_t)));
			} else
				gpu_storage->batch2_start = NULL;
			if (algo == LOCAL) {
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage->batch1_end),
								gpu_storage->gpu_max_n_alns * sizeof(uint32_t)));
				if (start == WITH_START) {
					CHECKCUDAERROR(
							cudaMalloc(&(gpu_storage->batch1_start),
									gpu_storage->gpu_max_n_alns * sizeof(uint32_t)));
				} else
					gpu_storage->batch1_start = NULL;
			} else {
				gpu_storage->batch1_start = NULL;
				gpu_storage->batch1_end = NULL;
			}
		}



	}
	//-------------------------------------------------------------------------------------------

	//------------------------copy sequence batches from CPU to GPU---------------------------
	CHECKCUDAERROR(cudaMemcpy(gpu_storage->unpacked1, batch1, actual_batch1_bytes, cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpy(gpu_storage->unpacked2, batch2, actual_batch2_bytes, cudaMemcpyHostToDevice));
	//----------------------------------------------------------------------------------------

    uint32_t BLOCKDIM = 128;
    uint32_t N_BLOCKS = (actual_n_alns + BLOCKDIM - 1) / BLOCKDIM;

    int batch1_tasks_per_thread = (int)ceil((double)actual_batch1_bytes/(8*BLOCKDIM*N_BLOCKS));
    int batch2_tasks_per_thread = (int)ceil((double)actual_batch2_bytes/(8*BLOCKDIM*N_BLOCKS));

    //launch packing kernel
    gasal_pack_kernel_4bit<<<N_BLOCKS, BLOCKDIM>>>((uint32_t*)(gpu_storage->unpacked1),
    						(uint32_t*)(gpu_storage->unpacked2), gpu_storage->packed1_4bit, gpu_storage->packed2_4bit,
    					    batch1_tasks_per_thread, batch2_tasks_per_thread, actual_batch1_bytes/4, actual_batch2_bytes/4);
    cudaError_t pack_kernel_err = cudaGetLastError();
    if ( cudaSuccess != pack_kernel_err )
    {
    	 fprintf(stderr, "Cuda error:%d(%s) at line no. %d in file %s\n", pack_kernel_err, cudaGetErrorString(pack_kernel_err), __LINE__, __FILE__);
         exit(EXIT_FAILURE);
    }

    //----------------------copy sequence offsets and lengths from CPU to GPU--------------------------------------
    CHECKCUDAERROR(cudaMemcpy(gpu_storage->lens1, batch1_lens, actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECKCUDAERROR(cudaMemcpy(gpu_storage->lens2, batch2_lens, actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECKCUDAERROR(cudaMemcpy(gpu_storage->offsets1, batch1_offsets, actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECKCUDAERROR(cudaMemcpy(gpu_storage->offsets2, batch2_offsets, actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice));
    //------------------------------------------------------------------------------------------------------------------------

    //--------------------------------------launch alignment kernels--------------------------------------------------------------
    if(algo == LOCAL) {
    	if (start == WITH_START) {
    		gasal_local_with_start_kernel<<<N_BLOCKS, BLOCKDIM>>>(gpu_storage->packed1_4bit, gpu_storage->packed2_4bit, gpu_storage->lens1,
    				gpu_storage->lens2, gpu_storage->offsets1, gpu_storage->offsets2, gpu_storage->aln_score,
    				gpu_storage->batch1_end, gpu_storage->batch2_end, gpu_storage->batch1_start,
    				gpu_storage->batch2_start, actual_n_alns);
    	} else {
    		gasal_local_kernel<<<N_BLOCKS, BLOCKDIM>>>(gpu_storage->packed1_4bit, gpu_storage->packed2_4bit, gpu_storage->lens1,
    				gpu_storage->lens2, gpu_storage->offsets1, gpu_storage->offsets2, gpu_storage->aln_score,
    				gpu_storage->batch1_end, gpu_storage->batch2_end, actual_n_alns);
    	}
    } else if (algo == SEMI_GLOBAL) {
    	if (start == WITH_START) {
    		gasal_semi_global_with_start_kernel<<<N_BLOCKS, BLOCKDIM>>>(gpu_storage->packed1_4bit, gpu_storage->packed2_4bit, gpu_storage->lens1,
    				gpu_storage->lens2, gpu_storage->offsets1, gpu_storage->offsets2, gpu_storage->aln_score, gpu_storage->batch2_end,
    				gpu_storage->batch2_start, actual_n_alns);
    	} else {
    		gasal_semi_global_kernel<<<N_BLOCKS, BLOCKDIM>>>(gpu_storage->packed1_4bit, gpu_storage->packed2_4bit, gpu_storage->lens1,
    				gpu_storage->lens2, gpu_storage->offsets1, gpu_storage->offsets2, gpu_storage->aln_score, gpu_storage->batch2_end,
    				actual_n_alns);
    	}

    } else if (algo == GLOBAL) {
    	gasal_global_kernel<<<N_BLOCKS, BLOCKDIM>>>(gpu_storage->packed1_4bit, gpu_storage->packed2_4bit, gpu_storage->lens1,
    			gpu_storage->lens2, gpu_storage->offsets1, gpu_storage->offsets2, gpu_storage->aln_score, actual_n_alns);
    }
    else {
    	fprintf(stderr, "Algo type invalid\n");
    	exit(EXIT_FAILURE);
    }
    //-----------------------------------------------------------------------------------------------------------------------
    cudaError_t aln_kernel_err = cudaGetLastError();
    if ( cudaSuccess != aln_kernel_err )
    {
    	fprintf(stderr, "Cuda error:%d(%s) at line no. %d in file %s\n", aln_kernel_err, cudaGetErrorString(aln_kernel_err), __LINE__, __FILE__);
    	exit(EXIT_FAILURE);
    }

    //------------------------copy alignment results from GPU to CPU--------------------------------------
    if (host_aln_score != NULL && gpu_storage->aln_score != NULL) CHECKCUDAERROR(cudaMemcpy(host_aln_score, gpu_storage->aln_score, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost));
    else {
    	fprintf(stderr, "The *host_aln_score input can't be NULL\n");
    	exit(EXIT_FAILURE);
    }
    if (host_batch1_start != NULL && gpu_storage->batch1_start != NULL) CHECKCUDAERROR(cudaMemcpy(host_batch1_start, gpu_storage->batch1_start, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost));
    if (host_batch2_start != NULL && gpu_storage->batch2_start != NULL) CHECKCUDAERROR(cudaMemcpy(host_batch2_start, gpu_storage->batch2_start, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost));
    if (host_batch1_end != NULL && gpu_storage->batch1_end != NULL) CHECKCUDAERROR(cudaMemcpy(host_batch1_end, gpu_storage->batch1_end, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost));
    if (host_batch2_end != NULL && gpu_storage->batch2_end != NULL) CHECKCUDAERROR(cudaMemcpy(host_batch2_end, gpu_storage->batch2_end, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost));
    //------------------------------------------------------------------------------------------------------

}


//GASAL2 asynchronous (a.k.a non-blocking) alignment function
void gasal_aln_async(gasal_gpu_storage_t *gpu_storage, const uint32_t actual_batch1_bytes, const uint32_t actual_batch2_bytes, const uint32_t actual_n_alns, int algo, int start) {

	cudaError_t err;
	if (actual_n_alns <= 0) {
		fprintf(stderr, "Must perform at least 1 alignment (n_alns > 0)\n");
		exit(EXIT_FAILURE);
	}
	if (actual_batch1_bytes <= 0) {
		fprintf(stderr, "Number of batch1_bytes should be greater than 0\n");
		exit(EXIT_FAILURE);
	}
	if (actual_batch2_bytes <= 0) {
		fprintf(stderr, "Number of batch2_bytes should be greater than 0\n");
		exit(EXIT_FAILURE);
	}

	if (actual_batch1_bytes % 8) {
		fprintf(stderr, "Number of batch1_bytes should be multiple of 8\n");
		exit(EXIT_FAILURE);
	}
	if (actual_batch2_bytes % 8) {
		fprintf(stderr, "Number of batch2_bytes should be multiple of 8\n");
		exit(EXIT_FAILURE);

	}

	//--------------if pre-allocated memory is less, allocate more--------------------------
	if (gpu_storage->gpu_max_batch1_bytes < actual_batch1_bytes) {
		fprintf(stderr, "max_batch1_bytes(%d) should be >= acutal_batch1_bytes(%d) \n", gpu_storage->gpu_max_batch1_bytes, actual_batch1_bytes);

		int i = 2;
		while ( (gpu_storage->gpu_max_batch1_bytes * i) < actual_batch1_bytes) i++;

		fprintf(stderr, "Therefore mallocing with max_batch1_bytes=%d \n", gpu_storage->gpu_max_batch1_bytes*i);
		gpu_storage->gpu_max_batch1_bytes = gpu_storage->gpu_max_batch1_bytes * i;

		if (gpu_storage->unpacked1 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->unpacked1));
		if (gpu_storage->packed1_4bit != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->packed1_4bit));

		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->unpacked1), gpu_storage->gpu_max_batch1_bytes * sizeof(uint8_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->packed1_4bit), (gpu_storage->gpu_max_batch1_bytes/8) * sizeof(uint32_t)));




	}

	if (gpu_storage->gpu_max_batch2_bytes < actual_batch2_bytes) {
		fprintf(stderr, "max_batch2_bytes(%d) should be >= acutal_batch2_bytes(%d) \n", gpu_storage->gpu_max_batch2_bytes, actual_batch2_bytes);

		int i = 2;
		while ( (gpu_storage->gpu_max_batch2_bytes * i) < actual_batch2_bytes) i++;

		fprintf(stderr, "Therefore mallocing with max_batch2_bytes=%d \n", gpu_storage->gpu_max_batch2_bytes*i);
		gpu_storage->gpu_max_batch2_bytes = gpu_storage->gpu_max_batch2_bytes * i;

		if (gpu_storage->unpacked2 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->unpacked2));
		if (gpu_storage->packed2_4bit != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->packed2_4bit));

		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->unpacked2), gpu_storage->gpu_max_batch2_bytes * sizeof(uint8_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->packed2_4bit), (gpu_storage->gpu_max_batch2_bytes/8) * sizeof(uint32_t)));


	}

	if (gpu_storage->gpu_max_n_alns < actual_n_alns) {
		fprintf(stderr, "Maximum possible number of alignment tasks(%d) should be >= acutal number of alignment tasks(%d) \n", gpu_storage->gpu_max_n_alns, actual_n_alns);

		int i = 2;
		while ( (gpu_storage->gpu_max_n_alns * i) < actual_n_alns) i++;

		fprintf(stderr, "Therefore mallocing with max_n_alns=%d \n", gpu_storage->gpu_max_n_alns*i);
		gpu_storage->gpu_max_n_alns = gpu_storage->gpu_max_n_alns * i;

		if (gpu_storage->offsets1 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->offsets1));
		if (gpu_storage->offsets2 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->offsets2));
		if (gpu_storage->lens1 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->lens1));
		if (gpu_storage->lens2 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->lens2));
		if (gpu_storage->aln_score != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->aln_score));
		if (gpu_storage->batch1_start != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->batch1_start));
		if (gpu_storage->batch2_start != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->batch2_start));
		if (gpu_storage->batch1_end != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->batch1_end));
		if (gpu_storage->batch2_end != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->batch2_end));

		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->lens1), gpu_storage->gpu_max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->lens2), gpu_storage->gpu_max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->offsets1), gpu_storage->gpu_max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->offsets2), gpu_storage->gpu_max_n_alns * sizeof(uint32_t)));

		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->aln_score), gpu_storage->gpu_max_n_alns * sizeof(int32_t)));
		if (algo == GLOBAL) {
			gpu_storage->batch1_start = NULL;
			gpu_storage->batch2_start = NULL;
			gpu_storage->batch1_end = NULL;
			gpu_storage->batch2_end = NULL;
		} else if (algo == SEMI_GLOBAL) {
			gpu_storage->batch1_start = NULL;
			gpu_storage->batch1_end = NULL;
			if (start == WITH_START) {
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage->batch2_start),
								gpu_storage->gpu_max_n_alns * sizeof(uint32_t)));
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage->batch2_end),
								gpu_storage->gpu_max_n_alns * sizeof(uint32_t)));
			} else {
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage->batch2_end),
								gpu_storage->gpu_max_n_alns * sizeof(uint32_t)));
				gpu_storage->batch2_start = NULL;
			}
		} else {
			if (start == WITH_START) {
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage->batch1_start),
								gpu_storage->gpu_max_n_alns * sizeof(uint32_t)));
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage->batch2_start),
								gpu_storage->gpu_max_n_alns * sizeof(uint32_t)));
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage->batch1_end),
								gpu_storage->gpu_max_n_alns * sizeof(uint32_t)));
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage->batch2_end),
								gpu_storage->gpu_max_n_alns * sizeof(uint32_t)));
			} else {
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage->batch1_end),
								gpu_storage->gpu_max_n_alns * sizeof(uint32_t)));
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage->batch2_end),
								gpu_storage->gpu_max_n_alns * sizeof(uint32_t)));
				gpu_storage->batch1_start = NULL;
				gpu_storage->batch2_start = NULL;
			}
		}



	}
	//------------------------------------------

	//------------------------launch copying of sequence batches from CPU to GPU---------------------------
	CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->unpacked1, gpu_storage->host_unpacked1, actual_batch1_bytes, cudaMemcpyHostToDevice, gpu_storage->str));
	CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->unpacked2, gpu_storage->host_unpacked2, actual_batch2_bytes, cudaMemcpyHostToDevice, gpu_storage->str));
	//-----------------------------------------------------------------------------------------------------------

    uint32_t BLOCKDIM = 128;
    uint32_t N_BLOCKS = (actual_n_alns + BLOCKDIM - 1) / BLOCKDIM;

    int batch1_tasks_per_thread = (int)ceil((double)actual_batch1_bytes/(8*BLOCKDIM*N_BLOCKS));
    int batch2_tasks_per_thread = (int)ceil((double)actual_batch2_bytes/(8*BLOCKDIM*N_BLOCKS));

    //-------------------------------------------launch packing kernel
    gasal_pack_kernel_4bit<<<N_BLOCKS, BLOCKDIM, 0, gpu_storage->str>>>((uint32_t*)(gpu_storage->unpacked1),
    						(uint32_t*)(gpu_storage->unpacked2), gpu_storage->packed1_4bit, gpu_storage->packed2_4bit,
    					    batch1_tasks_per_thread, batch2_tasks_per_thread, actual_batch1_bytes/4, actual_batch2_bytes/4);
    cudaError_t pack_kernel_err = cudaGetLastError();
    if ( cudaSuccess != pack_kernel_err )
    {
    	 fprintf(stderr, "Cuda error:%d(%s) at line no. %d in file %s\n", pack_kernel_err, cudaGetErrorString(pack_kernel_err), __LINE__, __FILE__);
         exit(EXIT_FAILURE);
    }

    //----------------------launch copying of sequence offsets and lengths from CPU to GPU--------------------------------------
    CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->lens1, gpu_storage->host_lens1, actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice, gpu_storage->str));
    CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->lens2, gpu_storage->host_lens2, actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice,  gpu_storage->str));
    CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->offsets1, gpu_storage->host_offsets1, actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice,  gpu_storage->str));
    CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->offsets2, gpu_storage->host_offsets2, actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice,  gpu_storage->str));
    //---------------------------------------------------------------------------------------------------------------

    //--------------------------------------launch alignment kernels--------------------------------------------------------------
        if(algo == LOCAL) {
        	if (start == WITH_START) {
        		gasal_local_with_start_kernel<<<N_BLOCKS, BLOCKDIM>>>(gpu_storage->packed1_4bit, gpu_storage->packed2_4bit, gpu_storage->lens1,
        				gpu_storage->lens2, gpu_storage->offsets1, gpu_storage->offsets2, gpu_storage->aln_score,
        				gpu_storage->batch1_end, gpu_storage->batch2_end, gpu_storage->batch1_start,
        				gpu_storage->batch2_start, actual_n_alns);
        	} else {
        		gasal_local_kernel<<<N_BLOCKS, BLOCKDIM>>>(gpu_storage->packed1_4bit, gpu_storage->packed2_4bit, gpu_storage->lens1,
        				gpu_storage->lens2, gpu_storage->offsets1, gpu_storage->offsets2, gpu_storage->aln_score,
        				gpu_storage->batch1_end, gpu_storage->batch2_end, actual_n_alns);
        	}
        } else if (algo == SEMI_GLOBAL) {
        	if (start == WITH_START) {
        		gasal_semi_global_with_start_kernel<<<N_BLOCKS, BLOCKDIM>>>(gpu_storage->packed1_4bit, gpu_storage->packed2_4bit, gpu_storage->lens1,
        				gpu_storage->lens2, gpu_storage->offsets1, gpu_storage->offsets2, gpu_storage->aln_score, gpu_storage->batch2_end,
        				gpu_storage->batch2_start, actual_n_alns);
        	} else {
        		gasal_semi_global_kernel<<<N_BLOCKS, BLOCKDIM>>>(gpu_storage->packed1_4bit, gpu_storage->packed2_4bit, gpu_storage->lens1,
        				gpu_storage->lens2, gpu_storage->offsets1, gpu_storage->offsets2, gpu_storage->aln_score, gpu_storage->batch2_end,
        				actual_n_alns);
        	}

        } else if (algo == GLOBAL) {
        	gasal_global_kernel<<<N_BLOCKS, BLOCKDIM>>>(gpu_storage->packed1_4bit, gpu_storage->packed2_4bit, gpu_storage->lens1,
        			gpu_storage->lens2, gpu_storage->offsets1, gpu_storage->offsets2, gpu_storage->aln_score, actual_n_alns);
        }
        else {
        	fprintf(stderr, "Algo type invalid\n");
        	exit(EXIT_FAILURE);
        }
        //-----------------------------------------------------------------------------------------------------------------------
    cudaError_t aln_kernel_err = cudaGetLastError();
    if ( cudaSuccess != aln_kernel_err )
    {
    	fprintf(stderr, "Cuda error:%d(%s) at line no. %d in file %s\n", aln_kernel_err, cudaGetErrorString(aln_kernel_err), __LINE__, __FILE__);
    	exit(EXIT_FAILURE);
    }

    //------------------------launch the copying of alignment results from GPU to CPU--------------------------------------
    if (gpu_storage->host_aln_score != NULL && gpu_storage->aln_score != NULL) CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->host_aln_score, gpu_storage->aln_score, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage->str));
    if (gpu_storage->host_batch1_start != NULL && gpu_storage->batch1_start != NULL) CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->host_batch1_start, gpu_storage->batch1_start, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage->str));
    if (gpu_storage->host_batch2_start != NULL && gpu_storage->batch2_start != NULL) CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->host_batch2_start, gpu_storage->batch2_start, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage->str));
    if (gpu_storage->host_batch1_end != NULL && gpu_storage->batch1_end != NULL) CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->host_batch1_end, gpu_storage->batch1_end, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage->str));
    if (gpu_storage->host_batch2_end != NULL && gpu_storage->batch2_end != NULL) CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->host_batch2_end, gpu_storage->batch2_end, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage->str));
    //-----------------------------------------------------------------------------------------------------------------------

    gpu_storage->is_free = 0; //set the availability of current stream to false

}




int gasal_is_aln_async_done(gasal_gpu_storage_t *gpu_storage) {

	cudaError_t err;
	err = cudaStreamQuery(gpu_storage->str);
	if (err != cudaSuccess ) {
		if (err == cudaErrorNotReady) return -1;
		else{
			fprintf(stderr, "Cuda error:%d(%s) at line no. %d in file %s\n", err, cudaGetErrorString(err), __LINE__, __FILE__);
			exit(EXIT_FAILURE);
		}
	}
	gpu_storage->is_free = 1;

	return 0;
}


void gasal_gpu_mem_alloc(gasal_gpu_storage_t *gpu_storage, int gpu_max_batch1_bytes, int gpu_max_batch2_bytes, int gpu_max_n_alns, int algo, int start) {

	cudaError_t err;
	if (gpu_storage->gpu_max_batch1_bytes % 8) {
		fprintf(stderr, "max_batch1_bytes should be multiple of 8\n");
		exit(EXIT_FAILURE);
	}
	if (gpu_storage->gpu_max_batch2_bytes % 8) {
		fprintf(stderr, "max_batch2_bytes should be multiple of 8\n");
		exit(EXIT_FAILURE);
	}
	CHECKCUDAERROR(cudaMalloc(&(gpu_storage->unpacked1), gpu_max_batch1_bytes * sizeof(uint8_t)));
	CHECKCUDAERROR(cudaMalloc(&(gpu_storage->unpacked2), gpu_max_batch2_bytes * sizeof(uint8_t)));

	CHECKCUDAERROR(cudaMalloc(&(gpu_storage->packed1_4bit), (gpu_max_batch1_bytes/8) * sizeof(uint32_t)));
	CHECKCUDAERROR(cudaMalloc(&(gpu_storage->packed2_4bit), (gpu_max_batch2_bytes/8) * sizeof(uint32_t)));

	CHECKCUDAERROR(cudaMalloc(&(gpu_storage->lens1), gpu_max_n_alns * sizeof(uint32_t)));
	CHECKCUDAERROR(cudaMalloc(&(gpu_storage->lens2), gpu_max_n_alns * sizeof(uint32_t)));
	CHECKCUDAERROR(cudaMalloc(&(gpu_storage->offsets1), gpu_max_n_alns * sizeof(uint32_t)));
	CHECKCUDAERROR(cudaMalloc(&(gpu_storage->offsets2), gpu_max_n_alns * sizeof(uint32_t)));

	CHECKCUDAERROR(cudaMalloc(&(gpu_storage->aln_score), gpu_max_n_alns * sizeof(int32_t)));
	if (algo == GLOBAL) {
		gpu_storage->batch1_start = NULL;
		gpu_storage->batch1_end = NULL;
		gpu_storage->batch2_start = NULL;
		gpu_storage->batch2_end = NULL;
	} else {
		CHECKCUDAERROR(
				cudaMalloc(&(gpu_storage->batch2_end),
						gpu_max_n_alns * sizeof(uint32_t)));
		if (start == WITH_START) {
			CHECKCUDAERROR(
					cudaMalloc(&(gpu_storage->batch2_start),
							gpu_max_n_alns * sizeof(uint32_t)));
		} else
			gpu_storage->batch2_start = NULL;
		if (algo == LOCAL) {
			CHECKCUDAERROR(
					cudaMalloc(&(gpu_storage->batch1_end),
							gpu_max_n_alns * sizeof(uint32_t)));
			if (start == WITH_START) {
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage->batch1_start),
								gpu_max_n_alns * sizeof(uint32_t)));
			} else
				gpu_storage->batch1_start = NULL;
		} else {
			gpu_storage->batch1_start = NULL;
			gpu_storage->batch1_end = NULL;
		}
	}

	gpu_storage->gpu_max_batch1_bytes = gpu_max_batch1_bytes;
	gpu_storage->gpu_max_batch2_bytes = gpu_max_batch2_bytes;
	gpu_storage->gpu_max_n_alns = gpu_max_n_alns;

}




void gasal_init_streams(gasal_gpu_storage_v *gpu_storage_vec, int host_max_batch1_bytes,  int gpu_max_batch1_bytes,  int host_max_batch2_bytes, int gpu_max_batch2_bytes, int host_max_n_alns, int gpu_max_n_alns, int algo, int start) {

	cudaError_t err;
	int i;
	for (i = 0; i < gpu_storage_vec->n; i++) {


		CHECKCUDAERROR(cudaMallocHost(&(gpu_storage_vec->a[i].host_unpacked1), host_max_batch1_bytes * sizeof(uint8_t)));
		CHECKCUDAERROR(cudaMallocHost(&(gpu_storage_vec->a[i].host_unpacked2), host_max_batch2_bytes * sizeof(uint8_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].unpacked1), gpu_max_batch1_bytes * sizeof(uint8_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].unpacked2), gpu_max_batch2_bytes * sizeof(uint8_t)));

		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].packed1_4bit), (gpu_max_batch1_bytes/8) * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].packed2_4bit), (gpu_max_batch2_bytes/8) * sizeof(uint32_t)));


		CHECKCUDAERROR(cudaMallocHost(&(gpu_storage_vec->a[i].host_lens1), host_max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMallocHost(&(gpu_storage_vec->a[i].host_lens2), host_max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMallocHost(&(gpu_storage_vec->a[i].host_offsets1), host_max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMallocHost(&(gpu_storage_vec->a[i].host_offsets2), host_max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMallocHost(&(gpu_storage_vec->a[i].host_aln_score), host_max_n_alns * sizeof(int32_t)));

		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].lens1), gpu_max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].lens2), gpu_max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].offsets1), gpu_max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].offsets2), gpu_max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].aln_score), gpu_max_n_alns * sizeof(int32_t)));
		if (algo == GLOBAL) {
			gpu_storage_vec->a[i].host_batch1_start = NULL;
			gpu_storage_vec->a[i].host_batch2_start = NULL;
			gpu_storage_vec->a[i].host_batch1_end = NULL;
			gpu_storage_vec->a[i].host_batch2_end = NULL;
			gpu_storage_vec->a[i].batch1_start = NULL;
			gpu_storage_vec->a[i].batch2_start = NULL;
			gpu_storage_vec->a[i].batch1_end = NULL;
			gpu_storage_vec->a[i].batch2_end = NULL;
		} else if (algo == SEMI_GLOBAL) {
			gpu_storage_vec->a[i].host_batch1_start = NULL;
			gpu_storage_vec->a[i].host_batch1_end = NULL;
			gpu_storage_vec->a[i].batch1_start = NULL;
			gpu_storage_vec->a[i].batch1_end = NULL;
			if (start == WITH_START) {
				CHECKCUDAERROR(cudaMallocHost(&(gpu_storage_vec->a[i].host_batch2_start),host_max_n_alns * sizeof(uint32_t)));
				CHECKCUDAERROR(cudaMallocHost(&(gpu_storage_vec->a[i].host_batch2_end),host_max_n_alns * sizeof(uint32_t)));

				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage_vec->a[i].batch2_start),
								gpu_max_n_alns * sizeof(uint32_t)));
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage_vec->a[i].batch2_end),
								gpu_max_n_alns * sizeof(uint32_t)));
			} else {
				CHECKCUDAERROR(cudaMallocHost(&(gpu_storage_vec->a[i].host_batch2_end),host_max_n_alns * sizeof(uint32_t)));
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage_vec->a[i].batch2_end),
								gpu_max_n_alns * sizeof(uint32_t)));
				gpu_storage_vec->a[i].host_batch2_start = NULL;
				gpu_storage_vec->a[i].batch2_start = NULL;
			}
		} else {
			if (start == WITH_START) {
				CHECKCUDAERROR(cudaMallocHost(&(gpu_storage_vec->a[i].host_batch1_start),host_max_n_alns * sizeof(uint32_t)));
				CHECKCUDAERROR(cudaMallocHost(&(gpu_storage_vec->a[i].host_batch2_start),host_max_n_alns * sizeof(uint32_t)));
				CHECKCUDAERROR(cudaMallocHost(&(gpu_storage_vec->a[i].host_batch1_end),host_max_n_alns * sizeof(uint32_t)));
				CHECKCUDAERROR(cudaMallocHost(&(gpu_storage_vec->a[i].host_batch2_end),host_max_n_alns * sizeof(uint32_t)));

				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage_vec->a[i].batch1_start),
								gpu_max_n_alns * sizeof(uint32_t)));
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage_vec->a[i].batch2_start),
								gpu_max_n_alns * sizeof(uint32_t)));
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage_vec->a[i].batch1_end),
								gpu_max_n_alns * sizeof(uint32_t)));
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage_vec->a[i].batch2_end),
								gpu_max_n_alns * sizeof(uint32_t)));
			} else {
				CHECKCUDAERROR(cudaMallocHost(&(gpu_storage_vec->a[i].host_batch1_end),host_max_n_alns * sizeof(uint32_t)));
				CHECKCUDAERROR(cudaMallocHost(&(gpu_storage_vec->a[i].host_batch2_end),host_max_n_alns * sizeof(uint32_t)));
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage_vec->a[i].batch1_end),
								gpu_max_n_alns * sizeof(uint32_t)));
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage_vec->a[i].batch2_end),
								gpu_max_n_alns * sizeof(uint32_t)));
				gpu_storage_vec->a[i].host_batch1_start = NULL;
				gpu_storage_vec->a[i].host_batch2_start = NULL;

				gpu_storage_vec->a[i].batch1_start = NULL;
				gpu_storage_vec->a[i].batch2_start = NULL;
			}
		}


		CHECKCUDAERROR(cudaStreamCreate(&(gpu_storage_vec->a[i].str)));
		gpu_storage_vec->a[i].is_free = 1;
		gpu_storage_vec->a[i].host_max_batch1_bytes = host_max_batch1_bytes;
		gpu_storage_vec->a[i].host_max_batch2_bytes = host_max_batch2_bytes;
		gpu_storage_vec->a[i].host_max_n_alns = host_max_n_alns;
		gpu_storage_vec->a[i].gpu_max_batch1_bytes = gpu_max_batch1_bytes;
		gpu_storage_vec->a[i].gpu_max_batch2_bytes = gpu_max_batch2_bytes;
		gpu_storage_vec->a[i].gpu_max_n_alns = gpu_max_n_alns;

	}


}

void gasal_gpu_mem_free(gasal_gpu_storage_t *gpu_storage) {

	cudaError_t err;

	if (gpu_storage->unpacked1 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->unpacked1));
	if (gpu_storage->unpacked2 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->unpacked2));
	if (gpu_storage->packed1_4bit != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->packed1_4bit));
	if (gpu_storage->packed2_4bit != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->packed2_4bit));
	if (gpu_storage->offsets1 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->offsets1));
	if (gpu_storage->offsets2 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->offsets2));
	if (gpu_storage->lens1 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->lens1));
	if (gpu_storage->lens2 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->lens2));
	if (gpu_storage->aln_score != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->aln_score));
	if (gpu_storage->batch1_start != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->batch1_start));
	if (gpu_storage->batch2_start != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->batch2_start));
	if (gpu_storage->batch1_end != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->batch1_end));
	if (gpu_storage->batch2_end != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->batch2_end));

}




void gasal_destroy_streams(gasal_gpu_storage_v *gpu_storage_vec) {

	cudaError_t err;

	int i;
	for (i = 0; i < gpu_storage_vec->n; i ++) {
		if (gpu_storage_vec->a[i].host_unpacked1 != NULL) CHECKCUDAERROR(cudaFreeHost(gpu_storage_vec->a[i].host_unpacked1));
		if (gpu_storage_vec->a[i].host_unpacked2 != NULL) CHECKCUDAERROR(cudaFreeHost(gpu_storage_vec->a[i].host_unpacked2));
		if (gpu_storage_vec->a[i].host_offsets1 != NULL) CHECKCUDAERROR(cudaFreeHost(gpu_storage_vec->a[i].host_offsets1));
		if (gpu_storage_vec->a[i].host_offsets2 != NULL) CHECKCUDAERROR(cudaFreeHost(gpu_storage_vec->a[i].host_offsets2));
		if (gpu_storage_vec->a[i].host_lens1 != NULL) CHECKCUDAERROR(cudaFreeHost(gpu_storage_vec->a[i].host_lens1));
		if (gpu_storage_vec->a[i].host_lens2 != NULL) CHECKCUDAERROR(cudaFreeHost(gpu_storage_vec->a[i].host_lens2));
		if (gpu_storage_vec->a[i].host_aln_score != NULL) CHECKCUDAERROR(cudaFreeHost(gpu_storage_vec->a[i].host_aln_score));
		if (gpu_storage_vec->a[i].host_batch1_start != NULL) CHECKCUDAERROR(cudaFreeHost(gpu_storage_vec->a[i].host_batch1_start));
		if (gpu_storage_vec->a[i].host_batch2_start != NULL) CHECKCUDAERROR(cudaFreeHost(gpu_storage_vec->a[i].host_batch2_start));
		if (gpu_storage_vec->a[i].host_batch1_end != NULL) CHECKCUDAERROR(cudaFreeHost(gpu_storage_vec->a[i].host_batch1_end));
		if (gpu_storage_vec->a[i].host_batch2_end != NULL) CHECKCUDAERROR(cudaFreeHost(gpu_storage_vec->a[i].host_batch2_end));

		if (gpu_storage_vec->a[i].unpacked1 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage_vec->a[i].unpacked1));
		if (gpu_storage_vec->a[i].unpacked2 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage_vec->a[i].unpacked2));
		if (gpu_storage_vec->a[i].packed1_4bit != NULL) CHECKCUDAERROR(cudaFree(gpu_storage_vec->a[i].packed1_4bit));
		if (gpu_storage_vec->a[i].packed2_4bit != NULL) CHECKCUDAERROR(cudaFree(gpu_storage_vec->a[i].packed2_4bit));
		if (gpu_storage_vec->a[i].offsets1 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage_vec->a[i].offsets1));
		if (gpu_storage_vec->a[i].offsets2 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage_vec->a[i].offsets2));
		if (gpu_storage_vec->a[i].lens1 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage_vec->a[i].lens1));
		if (gpu_storage_vec->a[i].lens2 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage_vec->a[i].lens2));
		if (gpu_storage_vec->a[i].aln_score != NULL) CHECKCUDAERROR(cudaFree(gpu_storage_vec->a[i].aln_score));
		if (gpu_storage_vec->a[i].batch1_start != NULL) CHECKCUDAERROR(cudaFree(gpu_storage_vec->a[i].batch1_start));
		if (gpu_storage_vec->a[i].batch2_start != NULL) CHECKCUDAERROR(cudaFree(gpu_storage_vec->a[i].batch2_start));
		if (gpu_storage_vec->a[i].batch1_end != NULL) CHECKCUDAERROR(cudaFree(gpu_storage_vec->a[i].batch1_end));
		if (gpu_storage_vec->a[i].batch2_end != NULL) CHECKCUDAERROR(cudaFree(gpu_storage_vec->a[i].batch2_end));

		if (gpu_storage_vec->a[i].str != NULL)CHECKCUDAERROR(cudaStreamDestroy(gpu_storage_vec->a[i].str));
	}



}


void gasal_destroy_gpu_storage_v(gasal_gpu_storage_v *gpu_storage_vec) {

	if(gpu_storage_vec->a != NULL) free(gpu_storage_vec->a);
}

void gasal_copy_subst_scores(gasal_subst_scores *subst){

	cudaError_t err;
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaGapO, &(subst->gap_open), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaGapExtend, &(subst->gap_extend), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	int32_t gapoe = subst->gap_open + subst->gap_extend;
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaGapOE, &(gapoe), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaMatchScore, &(subst->match), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaMismatchScore, &(subst->mismatch), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	return;
}




