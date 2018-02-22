#include "gasal.h"


enum system_type{
	HOST,
	GPU
};


#define CHECKCUDAERROR(error) \
		do{\
			err = error;\
			if (cudaSuccess != err ) { \
				fprintf(stderr, "Cuda error:%d(%s) at line no. %d in file %s\n", err, cudaGetErrorString(err), __LINE__, __FILE__); \
				exit(EXIT_FAILURE);\
			}\
		}while(0)\



//#define CUDASTREAMQUERYCHECK(error) \
//		do{\
//			err = error;\
//			if (cudaSuccess != err ) { \
//				if (err == cudaErrorNotReady) return -1; \
//				else{\
//					fprintf(stderr, "Cuda error:%d(%s) at line no. %d in file %s\n", err, cudaGetErrorString(err), __LINE__, __FILE__); \
//					exit(EXIT_FAILURE);\
//				}\
//			}\
//		}while(0)\

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


// The gasal local alignment function without start position computation

gasal_gpu_storage_v gasal_init_gpu_storage_v(int n_streams) {
	gasal_gpu_storage_v v;
	v.a = (gasal_gpu_storage_t*)calloc(n_streams, sizeof(gasal_gpu_storage_t));
	v.n = n_streams;
	return v;

}

void gasal_gpu_mem_alloc_contig (gasal_gpu_storage_t *gpu_storage, const uint32_t actual_batch1_bytes, const uint32_t actual_batch2_bytes, const uint32_t actual_n_alns, int algo, int start) {
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

		uint64_t req_gpu_malloc_size;



		if (algo == GLOBAL) {

			req_gpu_malloc_size = actual_batch1_bytes + actual_batch2_bytes  + (actual_batch1_bytes/2) + (actual_batch2_bytes/2) + (5 * actual_n_alns * sizeof(uint32_t));

			if (gpu_storage->is_gpu_mem_alloc && req_gpu_malloc_size > gpu_storage->max_gpu_malloc_size) {
				fprintf(stderr, "Required gpu malloc size (%llu bytes) > allocated gpu memory (%llu bytes)\n", req_gpu_malloc_size, gpu_storage->max_gpu_malloc_size);
				fprintf(stderr, "Remallocing %llu bytes of gpu memory\n", req_gpu_malloc_size);
				gpu_storage->max_gpu_malloc_size = req_gpu_malloc_size;
				cudaFree(gpu_storage->unpacked1);
				CHECKCUDAERROR( cudaMalloc(&(gpu_storage->unpacked1),
						gpu_storage->max_gpu_malloc_size));
			} else if (!gpu_storage->is_gpu_mem_alloc) {
				gpu_storage->max_gpu_malloc_size = req_gpu_malloc_size;
				CHECKCUDAERROR( cudaMalloc(&(gpu_storage->unpacked1),
						gpu_storage->max_gpu_malloc_size));

			}
			gpu_storage->unpacked2 = &(gpu_storage->unpacked1[actual_batch1_bytes]);
			gpu_storage->packed1_4bit = (uint32_t*)&(gpu_storage->unpacked2[actual_batch2_bytes]);
			gpu_storage->packed2_4bit = (uint32_t*)&(gpu_storage->packed1_4bit[actual_batch1_bytes/8]);
			gpu_storage->offsets1 = (uint32_t*)&(gpu_storage->packed2_4bit[actual_batch2_bytes/8]);
			gpu_storage->offsets2 = &(gpu_storage->offsets1[actual_n_alns]);
			gpu_storage->lens1 = &(gpu_storage->offsets2[actual_n_alns]);
			gpu_storage->lens2 = &(gpu_storage->lens1[actual_n_alns]);
			gpu_storage->aln_score = (int32_t*)&(gpu_storage->lens2[actual_n_alns]);
			gpu_storage->batch1_start = NULL;
			gpu_storage->batch2_start = NULL;
			gpu_storage->batch1_end = NULL;
			gpu_storage->batch2_end = NULL;



		} else if (algo == SEMI_GLOBAL){
			if (start == WITH_START) {
				req_gpu_malloc_size = actual_batch1_bytes + actual_batch2_bytes  + (actual_batch1_bytes/2) + (actual_batch2_bytes/2) + (7 * actual_n_alns * sizeof(uint32_t));

				if (gpu_storage->is_gpu_mem_alloc && req_gpu_malloc_size > gpu_storage->max_gpu_malloc_size) {
					fprintf(stderr, "Required gpu malloc size (%llu bytes) > allocated gpu memory (%llu bytes)", req_gpu_malloc_size, gpu_storage->max_gpu_malloc_size);
									fprintf(stderr, "Remallocing %llu bytes gpu memory", req_gpu_malloc_size);
					gpu_storage->max_gpu_malloc_size = req_gpu_malloc_size;
					cudaFree(gpu_storage->unpacked1);
					CHECKCUDAERROR( cudaMalloc(&(gpu_storage->unpacked1),
							gpu_storage->max_gpu_malloc_size));
				} else if (!gpu_storage->is_gpu_mem_alloc) {
					gpu_storage->max_gpu_malloc_size = req_gpu_malloc_size;
					CHECKCUDAERROR( cudaMalloc(&(gpu_storage->unpacked1),
							gpu_storage->max_gpu_malloc_size));

				}
				gpu_storage->unpacked2 = &(gpu_storage->unpacked1[actual_batch1_bytes]);
				gpu_storage->packed1_4bit = (uint32_t*)&(gpu_storage->unpacked2[actual_batch2_bytes]);
				gpu_storage->packed2_4bit = (uint32_t*)&(gpu_storage->packed1_4bit[actual_batch1_bytes/8]);
				gpu_storage->offsets1 = (uint32_t*)&(gpu_storage->packed2_4bit[actual_batch2_bytes/8]);
				gpu_storage->offsets2 = &(gpu_storage->offsets1[actual_n_alns]);
				gpu_storage->lens1 = &(gpu_storage->offsets2[actual_n_alns]);
				gpu_storage->lens2 = &(gpu_storage->lens1[actual_n_alns]);
				gpu_storage->aln_score = (int32_t*)&(gpu_storage->lens2[actual_n_alns]);
				gpu_storage->batch1_start = NULL;
				gpu_storage->batch2_start = &(gpu_storage->aln_score[actual_n_alns]);
				gpu_storage->batch1_end = NULL;
				gpu_storage->batch2_end = &(gpu_storage->batch2_start[actual_n_alns]);

			} else {
				req_gpu_malloc_size = actual_batch1_bytes + actual_batch2_bytes  + (actual_batch1_bytes/2) + (actual_batch2_bytes/2) + (6 * actual_n_alns * sizeof(uint32_t));

				if (gpu_storage->is_gpu_mem_alloc && req_gpu_malloc_size > gpu_storage->max_gpu_malloc_size) {
					fprintf(stderr, "Required gpu malloc size (%llu bytes) > allocated gpu memory (%llu bytes)\n", req_gpu_malloc_size, gpu_storage->max_gpu_malloc_size);
					fprintf(stderr, "Remallocing %llu bytes of gpu memory\n", req_gpu_malloc_size);
					gpu_storage->max_gpu_malloc_size = req_gpu_malloc_size;
					cudaFree(gpu_storage->unpacked1);
					CHECKCUDAERROR( cudaMalloc(&(gpu_storage->unpacked1),
							gpu_storage->max_gpu_malloc_size));
				} else if (!gpu_storage->is_gpu_mem_alloc) {
					gpu_storage->max_gpu_malloc_size = req_gpu_malloc_size;
					CHECKCUDAERROR( cudaMalloc(&(gpu_storage->unpacked1),
							gpu_storage->max_gpu_malloc_size));

				}
				gpu_storage->unpacked2 = &(gpu_storage->unpacked1[actual_batch1_bytes]);
				gpu_storage->packed1_4bit = (uint32_t*)&(gpu_storage->unpacked2[actual_batch2_bytes]);
				gpu_storage->packed2_4bit = (uint32_t*)&(gpu_storage->packed1_4bit[actual_batch1_bytes/8]);
				gpu_storage->offsets1 = (uint32_t*)&(gpu_storage->packed2_4bit[actual_batch2_bytes/8]);
				gpu_storage->offsets2 = &(gpu_storage->offsets1[actual_n_alns]);
				gpu_storage->lens1 = &(gpu_storage->offsets2[actual_n_alns]);
				gpu_storage->lens2 = &(gpu_storage->lens1[actual_n_alns]);
				gpu_storage->aln_score = (int32_t*)&(gpu_storage->lens2[actual_n_alns]);
				gpu_storage->batch1_start = NULL;
				gpu_storage->batch2_start =NULL;
				gpu_storage->batch1_end = NULL;
				gpu_storage->batch2_end = &(gpu_storage->aln_score[actual_n_alns]);

			}
		} else {
			if (start == WITH_START) {
				req_gpu_malloc_size = actual_batch1_bytes + actual_batch2_bytes  + (actual_batch1_bytes/2) + (actual_batch2_bytes/2) + (9 * actual_n_alns * sizeof(uint32_t));


				if (gpu_storage->is_gpu_mem_alloc && req_gpu_malloc_size > gpu_storage->max_gpu_malloc_size) {
					fprintf(stderr, "Required gpu malloc size (%llu bytes) > allocated gpu memory (%llu bytes)\n", req_gpu_malloc_size, gpu_storage->max_gpu_malloc_size);
					fprintf(stderr, "Remallocing %llu bytes of gpu memory\n", req_gpu_malloc_size);
					gpu_storage->max_gpu_malloc_size = req_gpu_malloc_size;
					cudaFree(gpu_storage->unpacked1);
					CHECKCUDAERROR( cudaMalloc(&(gpu_storage->unpacked1),
							gpu_storage->max_gpu_malloc_size));
				} else if (!gpu_storage->is_gpu_mem_alloc) {
					gpu_storage->max_gpu_malloc_size = req_gpu_malloc_size;
					CHECKCUDAERROR( cudaMalloc(&(gpu_storage->unpacked1),
							gpu_storage->max_gpu_malloc_size));

				}
				gpu_storage->unpacked2 = &(gpu_storage->unpacked1[actual_batch1_bytes]);
				gpu_storage->packed1_4bit = (uint32_t*)&(gpu_storage->unpacked2[actual_batch2_bytes]);
				gpu_storage->packed2_4bit = (uint32_t*)&(gpu_storage->packed1_4bit[actual_batch1_bytes/8]);
				gpu_storage->offsets1 = (uint32_t*)&(gpu_storage->packed2_4bit[actual_batch2_bytes/8]);
				gpu_storage->offsets2 = (uint32_t*)&(gpu_storage->offsets1[actual_n_alns]);
				gpu_storage->lens1 = (uint32_t*)&(gpu_storage->offsets2[actual_n_alns]);
				gpu_storage->lens2 = (uint32_t*)&(gpu_storage->lens1[actual_n_alns]);
				gpu_storage->aln_score = (int32_t*)&(gpu_storage->lens2[actual_n_alns]);
				gpu_storage->batch1_start = (int32_t*)&(gpu_storage->aln_score[actual_n_alns]);
				gpu_storage->batch2_start =(int32_t*)&(gpu_storage->batch1_start[actual_n_alns]);
				gpu_storage->batch1_end = (int32_t*)&(gpu_storage->batch2_start[actual_n_alns]);
				gpu_storage->batch2_end = (int32_t*)&(gpu_storage->batch1_end[actual_n_alns]);
			} else {
				req_gpu_malloc_size = actual_batch1_bytes + actual_batch2_bytes  + (actual_batch1_bytes/2) + (actual_batch2_bytes/2) + (7 * actual_n_alns * sizeof(uint32_t));
				if (gpu_storage->is_gpu_mem_alloc && req_gpu_malloc_size > gpu_storage->max_gpu_malloc_size) {
					fprintf(stderr, "Required gpu malloc size (%llu bytes) > allocated gpu memory (%llu bytes)\n", req_gpu_malloc_size, gpu_storage->max_gpu_malloc_size);
					fprintf(stderr, "Remallocing %llu bytes of gpu memory\n", req_gpu_malloc_size);
					gpu_storage->max_gpu_malloc_size = req_gpu_malloc_size;
					cudaFree(gpu_storage->unpacked1);
					CHECKCUDAERROR( cudaMalloc(&(gpu_storage->unpacked1),
							gpu_storage->max_gpu_malloc_size));
				} else if (!gpu_storage->is_gpu_mem_alloc) {
					gpu_storage->max_gpu_malloc_size = req_gpu_malloc_size;
					CHECKCUDAERROR( cudaMalloc(&(gpu_storage->unpacked1),
							gpu_storage->max_gpu_malloc_size));

				}
				gpu_storage->unpacked2 = &(gpu_storage->unpacked1[actual_batch1_bytes]);
				gpu_storage->packed1_4bit = (uint32_t*)&(gpu_storage->unpacked2[actual_batch2_bytes]);
				gpu_storage->packed2_4bit = (uint32_t*)&(gpu_storage->packed1_4bit[actual_batch1_bytes/8]);
				gpu_storage->offsets1 = (uint32_t*)&(gpu_storage->packed2_4bit[actual_batch2_bytes/8]);
				gpu_storage->offsets2 = &(gpu_storage->offsets1[actual_n_alns]);
				gpu_storage->lens1 = &(gpu_storage->offsets2[actual_n_alns]);
				gpu_storage->lens2 = &(gpu_storage->lens1[actual_n_alns]);
				gpu_storage->aln_score = (int32_t*)&(gpu_storage->lens2[actual_n_alns]);
				gpu_storage->batch1_start = NULL;
				gpu_storage->batch2_start =NULL;
				gpu_storage->batch1_end = &(gpu_storage->aln_score[actual_n_alns]);
				gpu_storage->batch2_end = &(gpu_storage->batch1_end[actual_n_alns]);
			}

		}
		gpu_storage->is_gpu_mem_alloc = 1;
		gpu_storage->is_gpu_mem_alloc_contig = 1;
}

/*void*/ gasal_gpu_storage_t* gasal_aln(const uint8_t *batch1, const uint32_t *batch1_offsets, const uint32_t *batch1_lens, const uint8_t *batch2, const uint32_t *batch2_offsets, const uint32_t *batch2_lens, const uint32_t batch1_bytes, const uint32_t batch2_bytes, const uint32_t n_alns, int32_t *host_aln_score, int32_t *host_batch1_start, int32_t *host_batch2_start, int32_t *host_batch1_end, int32_t *host_batch2_end, int algo, int start) {

	cudaError_t err;
	if (n_alns <= 0) {
		fprintf(stderr, "Must perform at least 1 alignment (n_alns > 0)\n");
		exit(EXIT_FAILURE);
	}
	if (batch1_bytes <= 0) {
		fprintf(stderr, "Number of batch1_bytes should be greater than 0\n");
		exit(EXIT_FAILURE);
	}
	if (batch2_bytes <= 0) {
		fprintf(stderr, "Number of batch2_bytes should be greater than 0\n");
		exit(EXIT_FAILURE);
	}

	if (batch1_bytes % 8) {
		fprintf(stderr, "Number of batch1_bytes should be multiple of 8\n");
		exit(EXIT_FAILURE);
	}
	if (batch2_bytes % 8) {
		fprintf(stderr, "Number of batch2_bytes should be multiple of 8\n");
		exit(EXIT_FAILURE);
	}
	if (algo != LOCAL && start != GLOBAL && start != SEMI_GLOBAL) {
		fprintf(stderr, "The value of  \"start\" parameter is not valid\n");
		exit(EXIT_FAILURE);
	}
	if (start != WITH_START && start != WITHOUT_START) {
		fprintf(stderr, "The value of  \"start\" parameter is not valid\n");
		exit(EXIT_FAILURE);
	}
	uint8_t *unpacked1, *unpacked2;
	uint32_t *packed1_4bit, *packed2_4bit;
	CHECKCUDAERROR(cudaMalloc(&unpacked1, batch1_bytes * sizeof(uint8_t)));
	CHECKCUDAERROR(cudaMalloc(&unpacked2, batch2_bytes * sizeof(uint8_t)));

	CHECKCUDAERROR(cudaMalloc(&packed1_4bit, (batch1_bytes/8) * sizeof(uint32_t)));
	CHECKCUDAERROR(cudaMalloc(&packed2_4bit, (batch2_bytes/8) * sizeof(uint32_t)));

	CHECKCUDAERROR(cudaMemcpy(unpacked1, batch1, batch1_bytes, cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpy(unpacked2, batch2, batch2_bytes, cudaMemcpyHostToDevice));


    uint32_t BLOCKDIM = 128;
    uint32_t N_BLOCKS = (n_alns + BLOCKDIM - 1) / BLOCKDIM;

    int batch1_tasks_per_thread = (int)ceil((double)batch1_bytes/(8*BLOCKDIM*N_BLOCKS));
    int batch2_tasks_per_thread = (int)ceil((double)batch2_bytes/(8*BLOCKDIM*N_BLOCKS));

    gasal_pack_kernel_4bit<<<N_BLOCKS, BLOCKDIM>>>((uint32_t*)(unpacked1),
    						(uint32_t*)(unpacked2), packed1_4bit, packed2_4bit,
    					    batch1_tasks_per_thread, batch2_tasks_per_thread, batch1_bytes/4, batch2_bytes/4);
    cudaError_t pack_kernel_err = cudaGetLastError();
    if ( cudaSuccess != pack_kernel_err )
    {
    	 fprintf(stderr, "Cuda error:%d(%s) at line no. %d in file %s\n", pack_kernel_err, cudaGetErrorString(pack_kernel_err), __LINE__, __FILE__);
         exit(EXIT_FAILURE);
    }

    uint32_t *lens1, *lens2, *offsets1, *offsets2;
    CHECKCUDAERROR(cudaMalloc(&lens1, n_alns * sizeof(uint32_t)));
    CHECKCUDAERROR(cudaMalloc(&lens2, n_alns * sizeof(uint32_t)));
    CHECKCUDAERROR(cudaMalloc(&offsets1, n_alns * sizeof(uint32_t)));
    CHECKCUDAERROR(cudaMalloc(&offsets2, n_alns * sizeof(uint32_t)));

    CHECKCUDAERROR(cudaMemcpy(lens1, batch1_lens, n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECKCUDAERROR(cudaMemcpy(lens2, batch2_lens, n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECKCUDAERROR(cudaMemcpy(offsets1, batch1_offsets, n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECKCUDAERROR(cudaMemcpy(offsets2, batch2_offsets, n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice));




    int32_t *aln_score, *batch1_start, *batch2_start, *batch1_end, *batch2_end;

    CHECKCUDAERROR(cudaMalloc(&aln_score, n_alns * sizeof(int32_t)));
    if (algo == GLOBAL) {
    	batch1_start = NULL;
    	batch1_end = NULL;
    	batch2_start = NULL;
    	batch2_end = NULL;
    } else {
    	CHECKCUDAERROR(cudaMalloc(&batch2_end, n_alns * sizeof(uint32_t)));
    	if (start == WITH_START) {
    		CHECKCUDAERROR(cudaMalloc(&batch2_start, n_alns * sizeof(uint32_t)));
    	} else
    		batch2_start = NULL;
    	if (algo == LOCAL) {
    		CHECKCUDAERROR(cudaMalloc(&batch1_end, n_alns * sizeof(uint32_t)));
    		if (start == WITH_START) {
    			CHECKCUDAERROR(cudaMalloc(&batch1_start, n_alns * sizeof(uint32_t)));
    		} else
    			batch1_start = NULL;
    	} else {
    		batch1_start = NULL;
    		batch1_end = NULL;
    	}
    }


    if (algo == LOCAL) {
    	if (start == WITH_START) {
    		gasal_local_with_start_kernel<<<N_BLOCKS, BLOCKDIM>>>(packed1_4bit, packed2_4bit, lens1,
    				lens2, offsets1, offsets2, aln_score,
    				batch1_end, batch2_end, batch1_start,
    				batch2_start, n_alns);
    	} else if (start == WITHOUT_START){
    		gasal_local_kernel<<<N_BLOCKS, BLOCKDIM>>>(packed1_4bit, packed2_4bit, lens1,
    				lens2, offsets1, offsets2, aln_score,
    				batch1_end, batch2_end, n_alns);
    	}

    } else if (algo ==  GLOBAL) {
    		gasal_global_kernel<<<N_BLOCKS, BLOCKDIM>>>(packed1_4bit, packed2_4bit, lens1,
    				lens2, offsets1, offsets2, aln_score, n_alns);
    } else {
    	if (start == WITH_START) {
    		gasal_semi_global_with_start_kernel<<<N_BLOCKS, BLOCKDIM>>>(packed1_4bit, packed2_4bit, lens1,
    				lens2, offsets1, offsets2, aln_score, batch2_end, batch2_start, n_alns);
    	} else if (start == WITHOUT_START){
    		gasal_semi_global_kernel<<<N_BLOCKS, BLOCKDIM>>>(packed1_4bit, packed2_4bit, lens1,
    				lens2, offsets1, offsets2, aln_score, batch2_end, n_alns);
    	}
    }

    cudaError_t aln_kernel_err = cudaGetLastError();
    if ( aln_kernel_err !=cudaSuccess  )
    {
    	fprintf(stderr, "Cuda error:%d(%s) at line no. %d in file %s\n", aln_kernel_err, cudaGetErrorString(aln_kernel_err), __LINE__, __FILE__);
    	exit(EXIT_FAILURE);
    }

    if (host_aln_score != NULL && aln_score != NULL) CHECKCUDAERROR(cudaMemcpy(host_aln_score, aln_score, n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost));
    else {
    		fprintf(stderr, "The *host_aln_score input can't be NULL I am here\n");
    		exit(EXIT_FAILURE);
    }
    if (host_batch1_start != NULL && batch1_start != NULL) CHECKCUDAERROR(cudaMemcpy(host_batch1_start, batch1_start, n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost));
    if (host_batch2_start != NULL && batch2_start != NULL) CHECKCUDAERROR(cudaMemcpy(host_batch2_start, batch2_start, n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost));
    if (host_batch1_end != NULL && batch1_end != NULL) CHECKCUDAERROR(cudaMemcpy(host_batch1_end, batch1_end, n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost));
    if (host_batch2_end != NULL && batch2_end != NULL) CHECKCUDAERROR(cudaMemcpy(host_batch2_end, batch2_end, n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost));

    gasal_gpu_storage_t*gpu_storage = (gasal_gpu_storage_t*)calloc(1, sizeof(gasal_gpu_storage_t));

    gpu_storage->unpacked1 = unpacked1;
    gpu_storage->unpacked2 = unpacked2;
    gpu_storage->packed1_4bit = packed1_4bit;
    gpu_storage->packed2_4bit = packed2_4bit;
    gpu_storage->offsets1 = offsets1;
    gpu_storage->offsets2 = offsets2;
    gpu_storage->lens1 = lens1;
    gpu_storage->lens2 = lens2;
    gpu_storage->aln_score = aln_score;
    gpu_storage->batch1_start = batch1_start;
    gpu_storage->batch2_start = batch2_start;
    gpu_storage->batch1_end = batch1_end;
    gpu_storage->batch2_end = batch2_end;


//    if (unpacked1 != NULL) CHECKCUDAERROR(cudaFree(unpacked1));
//    if (unpacked2 != NULL) CHECKCUDAERROR(cudaFree(unpacked2));
//    if (packed1_4bit != NULL) CHECKCUDAERROR(cudaFree(packed1_4bit));
//    if (packed2_4bit != NULL) CHECKCUDAERROR(cudaFree(packed2_4bit));
//    if (offsets1 != NULL) CHECKCUDAERROR(cudaFree(offsets1));
//    if (offsets2 != NULL) CHECKCUDAERROR(cudaFree(offsets2));
//    if (lens1 != NULL) CHECKCUDAERROR(cudaFree(lens1));
//    if (lens2 != NULL) CHECKCUDAERROR(cudaFree(lens2));
//    if (aln_score != NULL) CHECKCUDAERROR(cudaFree(aln_score));
//    if (batch1_start != NULL) CHECKCUDAERROR(cudaFree(batch1_start));
//    if (batch2_start != NULL) CHECKCUDAERROR(cudaFree(batch2_start));
//    if (batch1_end != NULL) CHECKCUDAERROR(cudaFree(batch1_end));
//    if (batch2_end != NULL) CHECKCUDAERROR(cudaFree(batch2_end));

    //return;
    return gpu_storage;
}

void gasal_aln_imp(const uint8_t *batch1, const uint32_t *batch1_offsets, const uint32_t *batch1_lens, const uint8_t *batch2, const uint32_t *batch2_offsets, const uint32_t *batch2_lens,   const uint32_t actual_batch1_bytes, const uint32_t actual_batch2_bytes, const uint32_t actual_n_alns, int32_t *host_aln_score, int32_t *host_batch1_start, int32_t *host_batch2_start, int32_t *host_batch1_end, int32_t *host_batch2_end,  int algo, int start, gasal_gpu_storage_t *gpu_storage) {

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
	if (gpu_storage->max_batch1_bytes < actual_batch1_bytes) {
		fprintf(stderr, "max_batch1_bytes(%d) should be >= acutal_batch1_bytes(%d) \n", gpu_storage->max_batch1_bytes, actual_batch1_bytes);

		int i = 2;
		while ( (gpu_storage->max_batch1_bytes * i) < actual_batch1_bytes) i++;

		fprintf(stderr, "Therefore mallocing with max_batch1_bytes=%d \n", gpu_storage->max_batch1_bytes*i);
		gpu_storage->max_batch1_bytes = gpu_storage->max_batch1_bytes * i;

		if (gpu_storage->unpacked1 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->unpacked1));
		if (gpu_storage->packed1_4bit != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->packed1_4bit));

		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->unpacked1), gpu_storage->max_batch1_bytes * sizeof(uint8_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->packed1_4bit), (gpu_storage->max_batch1_bytes/8) * sizeof(uint32_t)));




	}

	if (gpu_storage->max_batch2_bytes < actual_batch2_bytes) {
		fprintf(stderr, "max_batch2_bytes(%d) should be >= acutal_batch2_bytes(%d) \n", gpu_storage->max_batch2_bytes, actual_batch2_bytes);

		int i = 2;
		while ( (gpu_storage->max_batch2_bytes * i) < actual_batch2_bytes) i++;

		fprintf(stderr, "Therefore mallocing with max_batch2_bytes=%d \n", gpu_storage->max_batch2_bytes*i);
		gpu_storage->max_batch2_bytes = gpu_storage->max_batch2_bytes * i;

		if (gpu_storage->unpacked2 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->unpacked2));
		if (gpu_storage->packed2_4bit != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->packed2_4bit));

		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->unpacked2), gpu_storage->max_batch2_bytes * sizeof(uint8_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->packed2_4bit), (gpu_storage->max_batch2_bytes/8) * sizeof(uint32_t)));


	}

	if (gpu_storage->max_n_alns < actual_n_alns) {
		fprintf(stderr, "Maximum possible number of alignment tasks(%d) should be >= acutal number of alignment tasks(%d) \n", gpu_storage->max_n_alns, actual_n_alns);

		int i = 2;
		while ( (gpu_storage->max_n_alns * i) < actual_n_alns) i++;

		fprintf(stderr, "Therefore mallocing with max_n_alns=%d \n", gpu_storage->max_n_alns*i);
		gpu_storage->max_n_alns = gpu_storage->max_n_alns * i;

		if (gpu_storage->offsets1 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->offsets1));
		if (gpu_storage->offsets2 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->offsets2));
		if (gpu_storage->lens1 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->lens1));
		if (gpu_storage->lens2 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->lens2));
		if (gpu_storage->aln_score != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->aln_score));
		if (gpu_storage->batch1_start != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->batch1_start));
		if (gpu_storage->batch2_start != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->batch2_start));
		if (gpu_storage->batch1_end != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->batch1_end));
		if (gpu_storage->batch2_end != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->batch2_end));

		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->lens1), gpu_storage->max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->lens2), gpu_storage->max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->offsets1), gpu_storage->max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->offsets2), gpu_storage->max_n_alns * sizeof(uint32_t)));

		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->aln_score),gpu_storage->max_n_alns * sizeof(int32_t)));
		if (algo == GLOBAL) {
			gpu_storage->batch1_start = NULL;
			gpu_storage->batch1_end = NULL;
			gpu_storage->batch2_start = NULL;
			gpu_storage->batch2_end = NULL;
		} else {
			CHECKCUDAERROR(
					cudaMalloc(&(gpu_storage->batch2_end),
							gpu_storage->max_n_alns * sizeof(uint32_t)));
			if (start == WITH_START) {
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage->batch2_start),
								gpu_storage->max_n_alns * sizeof(uint32_t)));
			} else
				gpu_storage->batch2_start = NULL;
			if (algo == LOCAL) {
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage->batch1_end),
								gpu_storage->max_n_alns * sizeof(uint32_t)));
				if (start == WITH_START) {
					CHECKCUDAERROR(
							cudaMalloc(&(gpu_storage->batch1_start),
									gpu_storage->max_n_alns * sizeof(uint32_t)));
				} else
					gpu_storage->batch1_start = NULL;
			} else {
				gpu_storage->batch1_start = NULL;
				gpu_storage->batch1_end = NULL;
			}
		}



	}


	CHECKCUDAERROR(cudaMemcpy(gpu_storage->unpacked1, batch1, actual_batch1_bytes, cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpy(gpu_storage->unpacked2, batch2, actual_batch2_bytes, cudaMemcpyHostToDevice));


    uint32_t BLOCKDIM = 128;
    uint32_t N_BLOCKS = (actual_n_alns + BLOCKDIM - 1) / BLOCKDIM;

    int batch1_tasks_per_thread = (int)ceil((double)actual_batch1_bytes/(8*BLOCKDIM*N_BLOCKS));
    int batch2_tasks_per_thread = (int)ceil((double)actual_batch2_bytes/(8*BLOCKDIM*N_BLOCKS));

    gasal_pack_kernel_4bit<<<N_BLOCKS, BLOCKDIM>>>((uint32_t*)(gpu_storage->unpacked1),
    						(uint32_t*)(gpu_storage->unpacked2), gpu_storage->packed1_4bit, gpu_storage->packed2_4bit,
    					    batch1_tasks_per_thread, batch2_tasks_per_thread, actual_batch1_bytes/4, actual_batch2_bytes/4);
    cudaError_t pack_kernel_err = cudaGetLastError();
    if ( cudaSuccess != pack_kernel_err )
    {
    	 fprintf(stderr, "Cuda error:%d(%s) at line no. %d in file %s\n", pack_kernel_err, cudaGetErrorString(pack_kernel_err), __LINE__, __FILE__);
         exit(EXIT_FAILURE);
    }

    CHECKCUDAERROR(cudaMemcpy(gpu_storage->lens1, batch1_lens, actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECKCUDAERROR(cudaMemcpy(gpu_storage->lens2, batch2_lens, actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECKCUDAERROR(cudaMemcpy(gpu_storage->offsets1, batch1_offsets, actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CHECKCUDAERROR(cudaMemcpy(gpu_storage->offsets2, batch2_offsets, actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice));


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

    cudaError_t aln_kernel_err = cudaGetLastError();
    if ( cudaSuccess != aln_kernel_err )
    {
    	fprintf(stderr, "Cuda error:%d(%s) at line no. %d in file %s\n", aln_kernel_err, cudaGetErrorString(aln_kernel_err), __LINE__, __FILE__);
    	exit(EXIT_FAILURE);
    }

    if (host_aln_score != NULL && gpu_storage->aln_score != NULL) CHECKCUDAERROR(cudaMemcpy(host_aln_score, gpu_storage->aln_score, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost));
    else {
    	fprintf(stderr, "The *host_aln_score input can't be NULL\n");
    	exit(EXIT_FAILURE);
    }
    if (host_batch1_start != NULL && gpu_storage->batch1_start != NULL) CHECKCUDAERROR(cudaMemcpy(host_batch1_start, gpu_storage->batch1_start, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost));
    if (host_batch2_start != NULL && gpu_storage->batch2_start != NULL) CHECKCUDAERROR(cudaMemcpy(host_batch2_start, gpu_storage->batch2_start, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost));
    if (host_batch1_end != NULL && gpu_storage->batch1_end != NULL) CHECKCUDAERROR(cudaMemcpy(host_batch1_end, gpu_storage->batch1_end, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost));
    if (host_batch2_end != NULL && gpu_storage->batch2_end != NULL) CHECKCUDAERROR(cudaMemcpy(host_batch2_end, gpu_storage->batch2_end, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost));


}

gasal_gpu_storage_t* gasal_aln_async(const uint8_t *batch1, const uint32_t *batch1_offsets, const uint32_t *batch1_lens, const uint8_t *batch2, const uint32_t *batch2_offsets, const uint32_t *batch2_lens,   const uint32_t batch1_bytes, const uint32_t batch2_bytes, const uint32_t n_alns, int32_t *host_aln_score, int32_t *host_batch1_start, int32_t *host_batch2_start, int32_t *host_batch1_end, int32_t *host_batch2_end,   int algo, int start) {

	cudaError_t err;
	if (n_alns <= 0) {
		fprintf(stderr, "Must perform at least 1 alignment (n_alns > 0)\n");
		exit(EXIT_FAILURE);
	}
	if (batch1_bytes <= 0) {
		fprintf(stderr, "Number of batch1_bytes should be greater than 0\n");
		exit(EXIT_FAILURE);
	}
	if (batch2_bytes <= 0) {
		fprintf(stderr, "Number of batch2_bytes should be greater than 0\n");
		exit(EXIT_FAILURE);
	}

	if (batch1_bytes % 8) {
		fprintf(stderr, "Number of batch1_bytes should be multiple of 8\n");
		exit(EXIT_FAILURE);
	}
	if (batch2_bytes % 8) {
		fprintf(stderr, "Number of batch2_bytes should be multiple of 8\n");
		exit(EXIT_FAILURE);
	}


	gasal_gpu_storage_t *gpu_storage = (gasal_gpu_storage_t*)calloc(1, sizeof(gasal_gpu_storage_t));
	cudaStream_t str;

	CHECKCUDAERROR(cudaStreamCreate(&str));


	CHECKCUDAERROR(cudaMalloc(&(gpu_storage->unpacked1), batch1_bytes * sizeof(uint8_t)));
	CHECKCUDAERROR(cudaMalloc(&(gpu_storage->unpacked2), batch2_bytes * sizeof(uint8_t)));

	CHECKCUDAERROR(cudaMalloc(&(gpu_storage->packed1_4bit), (batch1_bytes/8) * sizeof(uint32_t)));
	CHECKCUDAERROR(cudaMalloc(&(gpu_storage->packed2_4bit), (batch2_bytes/8) * sizeof(uint32_t)));

	CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->unpacked1, batch1, batch1_bytes, cudaMemcpyHostToDevice, str));
	CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->unpacked2, batch2, batch2_bytes, cudaMemcpyHostToDevice, str));


    uint32_t BLOCKDIM = 128;
    uint32_t N_BLOCKS = (n_alns + BLOCKDIM - 1) / BLOCKDIM;

    int batch1_tasks_per_thread = (int)ceil((double)batch1_bytes/(8*BLOCKDIM*N_BLOCKS));
    int batch2_tasks_per_thread = (int)ceil((double)batch2_bytes/(8*BLOCKDIM*N_BLOCKS));

    gasal_pack_kernel_4bit<<<N_BLOCKS, BLOCKDIM, 0, str>>>((uint32_t*)(gpu_storage->unpacked1),
    						(uint32_t*)(gpu_storage->unpacked2), gpu_storage->packed1_4bit, gpu_storage->packed2_4bit,
    					    batch1_tasks_per_thread, batch2_tasks_per_thread, batch1_bytes/4, batch2_bytes/4);
    cudaError_t pack_kernel_err = cudaGetLastError();
    if ( cudaSuccess != pack_kernel_err )
    {
    	 fprintf(stderr, "Cuda error:%d(%s) at line no. %d in file %s\n", pack_kernel_err, cudaGetErrorString(pack_kernel_err), __LINE__, __FILE__);
         exit(EXIT_FAILURE);
    }


    CHECKCUDAERROR(cudaMalloc(&(gpu_storage->lens1), n_alns * sizeof(uint32_t)));
    CHECKCUDAERROR(cudaMalloc(&(gpu_storage->lens2), n_alns * sizeof(uint32_t)));
    CHECKCUDAERROR(cudaMalloc(&(gpu_storage->offsets1), n_alns * sizeof(uint32_t)));
    CHECKCUDAERROR(cudaMalloc(&(gpu_storage->offsets2), n_alns * sizeof(uint32_t)));

    CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->lens1, batch1_lens, n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice, str));
    CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->lens2, batch2_lens, n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice, str));
    CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->offsets1, batch1_offsets, n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice, str));
    CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->offsets2, batch2_offsets, n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice, str));





	CHECKCUDAERROR(cudaMalloc(&(gpu_storage->aln_score), n_alns * sizeof(int32_t)));
	if (algo == GLOBAL) {
		gpu_storage->batch1_start = NULL;
		gpu_storage->batch1_end = NULL;
		gpu_storage->batch2_start = NULL;
		gpu_storage->batch2_end = NULL;
	} else {
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->batch2_end), n_alns * sizeof(uint32_t)));
		if (start == WITH_START){
			CHECKCUDAERROR(cudaMalloc(&(gpu_storage->batch2_start), n_alns * sizeof(uint32_t)));
		}
		else gpu_storage->batch2_start = NULL;
		if (algo == LOCAL) {
			CHECKCUDAERROR(cudaMalloc(&(gpu_storage->batch1_end), n_alns * sizeof(uint32_t)));
			if (start == WITH_START){
				CHECKCUDAERROR(cudaMalloc(&(gpu_storage->batch1_start), n_alns * sizeof(uint32_t)));
			}
			else gpu_storage->batch1_start = NULL;
		} else {
			gpu_storage->batch1_start = NULL;
			gpu_storage->batch1_end = NULL;
		}
	}



    if (start == WITH_START) {
		gasal_local_with_start_kernel<<<N_BLOCKS, BLOCKDIM, 0, str>>>(gpu_storage->packed1_4bit, gpu_storage->packed2_4bit, gpu_storage->lens1,
				gpu_storage->lens2, gpu_storage->offsets1, gpu_storage->offsets2, gpu_storage->aln_score,
				gpu_storage->batch1_end, gpu_storage->batch2_end, gpu_storage->batch1_start,
				gpu_storage->batch2_start, n_alns);
	} else {
		gasal_local_kernel<<<N_BLOCKS, BLOCKDIM, 0, str>>>(gpu_storage->packed1_4bit, gpu_storage->packed2_4bit, gpu_storage->lens1,
				gpu_storage->lens2, gpu_storage->offsets1, gpu_storage->offsets2, gpu_storage->aln_score,
				gpu_storage->batch1_end, gpu_storage->batch2_end, n_alns);
	}

    cudaError_t aln_kernel_err = cudaGetLastError();
    if ( cudaSuccess != aln_kernel_err )
    {
    	fprintf(stderr, "Cuda error:%d(%s) at line no. %d in file %s\n", aln_kernel_err, cudaGetErrorString(aln_kernel_err), __LINE__, __FILE__);
    	exit(EXIT_FAILURE);
    }



    if (host_aln_score != NULL && gpu_storage->aln_score != NULL) CHECKCUDAERROR(cudaMemcpyAsync(host_aln_score, gpu_storage->aln_score, n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, str));
    else {
    	fprintf(stderr, "The *host_aln_score input can't be NULL\n");
    	exit(EXIT_FAILURE);
    }
    if (host_batch1_start != NULL && gpu_storage->batch1_start != NULL) CHECKCUDAERROR(cudaMemcpyAsync(host_batch1_start, gpu_storage->batch1_start, n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, str));
    if (host_batch2_start != NULL && gpu_storage->batch2_start != NULL) CHECKCUDAERROR(cudaMemcpyAsync(host_batch2_start, gpu_storage->batch2_start, n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, str));
    if (host_batch1_end != NULL && gpu_storage->batch1_end != NULL) CHECKCUDAERROR(cudaMemcpyAsync(host_batch1_end, gpu_storage->batch1_end, n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, str));
    if (host_batch2_end != NULL && gpu_storage->batch2_end != NULL) CHECKCUDAERROR(cudaMemcpyAsync(host_batch2_end, gpu_storage->batch2_end, n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, str));

//    gpu_storage->host_aln_score = host_aln_score;
//    gpu_storage->host_batch1_start = host_batch1_start;
//    gpu_storage->host_batch2_start = host_batch2_start;
//    gpu_storage->host_batch1_end = host_batch1_end;
//    gpu_storage->host_batch2_end = host_batch2_end;
    gpu_storage->n_alns = n_alns;
    gpu_storage->str = str;


    return gpu_storage;
}

void gasal_aln_async_new(gasal_gpu_storage_t *gpu_storage, const uint32_t actual_batch1_bytes, const uint32_t actual_batch2_bytes, const uint32_t actual_n_alns, int algo, int start) {

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
	if (gpu_storage->max_batch1_bytes < actual_batch1_bytes) {
		fprintf(stderr, "max_batch1_bytes(%d) should be >= acutal_batch1_bytes(%d) \n", gpu_storage->max_batch1_bytes, actual_batch1_bytes);

		int i = 2;
		while ( (gpu_storage->max_batch1_bytes * i) < actual_batch1_bytes) i++;

		fprintf(stderr, "Therefore mallocing with max_batch1_bytes=%d \n", gpu_storage->max_batch1_bytes*i);
		gpu_storage->max_batch1_bytes = gpu_storage->max_batch1_bytes * i;

		if (gpu_storage->unpacked1 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->unpacked1));
		if (gpu_storage->packed1_4bit != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->packed1_4bit));

		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->unpacked1), gpu_storage->max_batch1_bytes * sizeof(uint8_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->packed1_4bit), (gpu_storage->max_batch1_bytes/8) * sizeof(uint32_t)));




	}

	if (gpu_storage->max_batch2_bytes < actual_batch2_bytes) {
		fprintf(stderr, "max_batch2_bytes(%d) should be >= acutal_batch2_bytes(%d) \n", gpu_storage->max_batch2_bytes, actual_batch2_bytes);

		int i = 2;
		while ( (gpu_storage->max_batch2_bytes * i) < actual_batch2_bytes) i++;

		fprintf(stderr, "Therefore mallocing with max_batch2_bytes=%d \n", gpu_storage->max_batch2_bytes*i);
		gpu_storage->max_batch2_bytes = gpu_storage->max_batch2_bytes * i;

		if (gpu_storage->unpacked2 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->unpacked2));
		if (gpu_storage->packed2_4bit != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->packed2_4bit));

		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->unpacked2), gpu_storage->max_batch2_bytes * sizeof(uint8_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->packed2_4bit), (gpu_storage->max_batch2_bytes/8) * sizeof(uint32_t)));


	}

	if (gpu_storage->max_n_alns < actual_n_alns) {
		fprintf(stderr, "Maximum possible number of alignment tasks(%d) should be >= acutal number of alignment tasks(%d) \n", gpu_storage->max_n_alns, actual_n_alns);

		int i = 2;
		while ( (gpu_storage->max_n_alns * i) < actual_n_alns) i++;

		fprintf(stderr, "Therefore mallocing with max_n_alns=%d \n", gpu_storage->max_n_alns*i);
		gpu_storage->max_n_alns = gpu_storage->max_n_alns * i;

		if (gpu_storage->offsets1 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->offsets1));
		if (gpu_storage->offsets2 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->offsets2));
		if (gpu_storage->lens1 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->lens1));
		if (gpu_storage->lens2 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->lens2));
		if (gpu_storage->aln_score != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->aln_score));
		if (gpu_storage->batch1_start != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->batch1_start));
		if (gpu_storage->batch2_start != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->batch2_start));
		if (gpu_storage->batch1_end != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->batch1_end));
		if (gpu_storage->batch2_end != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->batch2_end));

		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->lens1), gpu_storage->max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->lens2), gpu_storage->max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->offsets1), gpu_storage->max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->offsets2), gpu_storage->max_n_alns * sizeof(uint32_t)));

		CHECKCUDAERROR(cudaMalloc(&(gpu_storage->aln_score), gpu_storage->max_n_alns * sizeof(int32_t)));
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
								gpu_storage->max_n_alns * sizeof(uint32_t)));
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage->batch2_end),
								gpu_storage->max_n_alns * sizeof(uint32_t)));
			} else {
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage->batch2_end),
								gpu_storage->max_n_alns * sizeof(uint32_t)));
				gpu_storage->batch2_start = NULL;
			}
		} else {
			if (start == WITH_START) {
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage->batch1_start),
								gpu_storage->max_n_alns * sizeof(uint32_t)));
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage->batch2_start),
								gpu_storage->max_n_alns * sizeof(uint32_t)));
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage->batch1_end),
								gpu_storage->max_n_alns * sizeof(uint32_t)));
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage->batch2_end),
								gpu_storage->max_n_alns * sizeof(uint32_t)));
			} else {
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage->batch1_end),
								gpu_storage->max_n_alns * sizeof(uint32_t)));
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage->batch2_end),
								gpu_storage->max_n_alns * sizeof(uint32_t)));
				gpu_storage->batch1_start = NULL;
				gpu_storage->batch2_start = NULL;
			}
		}



	}


	CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->unpacked1, gpu_storage->host_unpacked1, actual_batch1_bytes, cudaMemcpyHostToDevice, gpu_storage->str));
	CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->unpacked2, gpu_storage->host_unpacked2, actual_batch2_bytes, cudaMemcpyHostToDevice, gpu_storage->str));


    uint32_t BLOCKDIM = 128;
    uint32_t N_BLOCKS = (actual_n_alns + BLOCKDIM - 1) / BLOCKDIM;

    int batch1_tasks_per_thread = (int)ceil((double)actual_batch1_bytes/(8*BLOCKDIM*N_BLOCKS));
    int batch2_tasks_per_thread = (int)ceil((double)actual_batch2_bytes/(8*BLOCKDIM*N_BLOCKS));

    gasal_pack_kernel_4bit<<<N_BLOCKS, BLOCKDIM, 0, gpu_storage->str>>>((uint32_t*)(gpu_storage->unpacked1),
    						(uint32_t*)(gpu_storage->unpacked2), gpu_storage->packed1_4bit, gpu_storage->packed2_4bit,
    					    batch1_tasks_per_thread, batch2_tasks_per_thread, actual_batch1_bytes/4, actual_batch2_bytes/4);
    cudaError_t pack_kernel_err = cudaGetLastError();
    if ( cudaSuccess != pack_kernel_err )
    {
    	 fprintf(stderr, "Cuda error:%d(%s) at line no. %d in file %s\n", pack_kernel_err, cudaGetErrorString(pack_kernel_err), __LINE__, __FILE__);
         exit(EXIT_FAILURE);
    }

    CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->lens1, gpu_storage->host_lens1, actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice, gpu_storage->str));
    CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->lens2, gpu_storage->host_lens2, actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice,  gpu_storage->str));
    CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->offsets1, gpu_storage->host_offsets1, actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice,  gpu_storage->str));
    CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->offsets2, gpu_storage->host_offsets2, actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice,  gpu_storage->str));


    if (start == WITH_START) {
		gasal_local_with_start_kernel<<<N_BLOCKS, BLOCKDIM, 0, gpu_storage->str>>>(gpu_storage->packed1_4bit, gpu_storage->packed2_4bit, gpu_storage->lens1,
				gpu_storage->lens2, gpu_storage->offsets1, gpu_storage->offsets2, gpu_storage->aln_score,
				gpu_storage->batch1_end, gpu_storage->batch2_end, gpu_storage->batch1_start,
				gpu_storage->batch2_start, actual_n_alns);
	} else {
		gasal_local_kernel<<<N_BLOCKS, BLOCKDIM, 0, gpu_storage->str>>>(gpu_storage->packed1_4bit, gpu_storage->packed2_4bit, gpu_storage->lens1,
				gpu_storage->lens2, gpu_storage->offsets1, gpu_storage->offsets2, gpu_storage->aln_score,
				gpu_storage->batch1_end, gpu_storage->batch2_end, actual_n_alns);
	}

    cudaError_t aln_kernel_err = cudaGetLastError();
    if ( cudaSuccess != aln_kernel_err )
    {
    	fprintf(stderr, "Cuda error:%d(%s) at line no. %d in file %s\n", aln_kernel_err, cudaGetErrorString(aln_kernel_err), __LINE__, __FILE__);
    	exit(EXIT_FAILURE);
    }

    if(gpu_storage->is_gpu_mem_alloc_contig == 1 && gpu_storage->is_host_mem_alloc_contig == 1) CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->host_aln_score, gpu_storage->aln_score, 5 * actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage->str));
    else {
    	if (gpu_storage->host_aln_score != NULL && gpu_storage->aln_score != NULL) CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->host_aln_score, gpu_storage->aln_score, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage->str));
    	if (gpu_storage->host_batch1_start != NULL && gpu_storage->batch1_start != NULL) CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->host_batch1_start, gpu_storage->batch1_start, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage->str));
    	if (gpu_storage->host_batch2_start != NULL && gpu_storage->batch2_start != NULL) CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->host_batch2_start, gpu_storage->batch2_start, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage->str));
    	if (gpu_storage->host_batch1_end != NULL && gpu_storage->batch1_end != NULL) CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->host_batch1_end, gpu_storage->batch1_end, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage->str));
    	if (gpu_storage->host_batch2_end != NULL && gpu_storage->batch2_end != NULL) CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->host_batch2_end, gpu_storage->batch2_end, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage->str));
    }

    gpu_storage->n_alns = actual_n_alns;
    gpu_storage->is_free = 0;

}

void gasal_aln_async_new2(gasal_gpu_storage_t *gpu_storage, const uint32_t actual_batch1_bytes, const uint32_t actual_batch2_bytes, const uint32_t actual_n_alns, int algo, int start) {

	cudaError_t err;

	//cudaStream_t str;
	CHECKCUDAERROR(cudaStreamCreate(&(gpu_storage->str)));
	//gpu_storage->str = str;

	gpu_storage->n_alns = actual_n_alns;

	CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->unpacked1, gpu_storage->host_unpacked1, actual_batch1_bytes, cudaMemcpyHostToDevice, gpu_storage->str));
	CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->unpacked2, gpu_storage->host_unpacked2, actual_batch2_bytes, cudaMemcpyHostToDevice, gpu_storage->str));


    uint32_t BLOCKDIM = 128;
    uint32_t N_BLOCKS = (actual_n_alns + BLOCKDIM - 1) / BLOCKDIM;

    int batch1_tasks_per_thread = (int)ceil((double)actual_batch1_bytes/(8*BLOCKDIM*N_BLOCKS));
    int batch2_tasks_per_thread = (int)ceil((double)actual_batch2_bytes/(8*BLOCKDIM*N_BLOCKS));

    gasal_pack_kernel_4bit<<<N_BLOCKS, BLOCKDIM, 0, gpu_storage->str>>>((uint32_t*)(gpu_storage->unpacked1),
    						(uint32_t*)(gpu_storage->unpacked2), gpu_storage->packed1_4bit, gpu_storage->packed2_4bit,
    					    batch1_tasks_per_thread, batch2_tasks_per_thread, actual_batch1_bytes/4, actual_batch2_bytes/4);
//    cudaError_t pack_kernel_err = cudaGetLastError();
//    if ( cudaSuccess != pack_kernel_err )
//    {
//    	 fprintf(stderr, "Cuda error:%d(%s) at line no. %d in file %s\n", pack_kernel_err, cudaGetErrorString(pack_kernel_err), __LINE__, __FILE__);
//         exit(EXIT_FAILURE);
//    }
    CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->lens1, gpu_storage->host_lens1, actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice, gpu_storage->str));
    CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->lens2, gpu_storage->host_lens2, actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice,  gpu_storage->str));
    CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->offsets1, gpu_storage->host_offsets1, actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice,  gpu_storage->str));
    CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->offsets2, gpu_storage->host_offsets2, actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice,  gpu_storage->str));


    if (start == WITH_START) {
		gasal_local_with_start_kernel<<<N_BLOCKS, BLOCKDIM, 0, gpu_storage->str>>>(gpu_storage->packed1_4bit, gpu_storage->packed2_4bit, gpu_storage->lens1,
				gpu_storage->lens2, gpu_storage->offsets1, gpu_storage->offsets2, gpu_storage->aln_score,
				gpu_storage->batch1_end, gpu_storage->batch2_end, gpu_storage->batch1_start,
				gpu_storage->batch2_start, actual_n_alns);
	} else {
		gasal_local_kernel<<<N_BLOCKS, BLOCKDIM, 0, gpu_storage->str>>>(gpu_storage->packed1_4bit, gpu_storage->packed2_4bit, gpu_storage->lens1,
				gpu_storage->lens2, gpu_storage->offsets1, gpu_storage->offsets2, gpu_storage->aln_score,
				gpu_storage->batch1_end, gpu_storage->batch2_end, actual_n_alns);
	}



//    cudaError_t aln_kernel_err = cudaGetLastError();
//    if ( cudaSuccess != aln_kernel_err )
//    {
//    	fprintf(stderr, "Cuda error:%d(%s) at line no. %d in file %s\n", aln_kernel_err, cudaGetErrorString(aln_kernel_err), __LINE__, __FILE__);
//    	exit(EXIT_FAILURE);
//    }
////
//        if (gpu_storage->host_aln_score != NULL && gpu_storage->aln_score != NULL) CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->host_aln_score, gpu_storage->aln_score, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage->str));
//        else {
//        	fprintf(stderr, "The *host_aln_score input can't be NULL\n");
//        	exit(EXIT_FAILURE);
//        }
      //if (gpu_storage->host_results != NULL && gpu_storage->aln_score != NULL) CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->host_results, gpu_storage->aln_score, 5 * actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage->str));
//    if (gpu_storage->host_batch1_start != NULL && gpu_storage->batch1_start != NULL) CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->host_batch1_start, gpu_storage->batch1_start, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage->str));
//    if (gpu_storage->host_batch2_start != NULL && gpu_storage->batch2_start != NULL) CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->host_batch2_start, gpu_storage->batch2_start, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage->str));
//    if (gpu_storage->host_batch1_end != NULL && gpu_storage->batch1_end != NULL) CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->host_batch1_end, gpu_storage->batch1_end, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost,gpu_storage->str));
//    if (gpu_storage->host_batch2_end != NULL && gpu_storage->batch2_end != NULL) CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->host_batch2_end, gpu_storage->batch2_end, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage->str));
}

void gasal_aln_async_new3(gasal_gpu_storage_t *gpu_storage, const uint32_t actual_batch1_bytes, const uint32_t actual_batch2_bytes, const uint32_t actual_n_alns, int algo, int start) {

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

	gpu_storage->is_gpu_mem_alloc = 1;
	gasal_gpu_mem_alloc_contig(gpu_storage, actual_batch1_bytes, actual_batch2_bytes, actual_n_alns, algo, start);


	CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->unpacked1, gpu_storage->host_unpacked1, actual_batch1_bytes, cudaMemcpyHostToDevice, gpu_storage->str));
	CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->unpacked2, gpu_storage->host_unpacked2, actual_batch2_bytes, cudaMemcpyHostToDevice, gpu_storage->str));


    uint32_t BLOCKDIM = 128;
    uint32_t N_BLOCKS = (actual_n_alns + BLOCKDIM - 1) / BLOCKDIM;

    int batch1_tasks_per_thread = (int)ceil((double)actual_batch1_bytes/(8*BLOCKDIM*N_BLOCKS));
    int batch2_tasks_per_thread = (int)ceil((double)actual_batch2_bytes/(8*BLOCKDIM*N_BLOCKS));

    gasal_pack_kernel_4bit<<<N_BLOCKS, BLOCKDIM, 0, gpu_storage->str>>>((uint32_t*)(gpu_storage->unpacked1),
    						(uint32_t*)(gpu_storage->unpacked2), gpu_storage->packed1_4bit, gpu_storage->packed2_4bit,
    					    batch1_tasks_per_thread, batch2_tasks_per_thread, actual_batch1_bytes/4, actual_batch2_bytes/4);
    cudaError_t pack_kernel_err = cudaGetLastError();
    if ( cudaSuccess != pack_kernel_err )
    {
    	 fprintf(stderr, "Cuda error:%d(%s) at line no. %d in file %s\n", pack_kernel_err, cudaGetErrorString(pack_kernel_err), __LINE__, __FILE__);
         exit(EXIT_FAILURE);
    }

    CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->lens1, gpu_storage->host_lens1, actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice, gpu_storage->str));
    CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->lens2, gpu_storage->host_lens2, actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice,  gpu_storage->str));
    CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->offsets1, gpu_storage->host_offsets1, actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice,  gpu_storage->str));
    CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage->offsets2, gpu_storage->host_offsets2, actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice,  gpu_storage->str));


    if (start == WITH_START) {
		gasal_local_with_start_kernel<<<N_BLOCKS, BLOCKDIM, 0, gpu_storage->str>>>(gpu_storage->packed1_4bit, gpu_storage->packed2_4bit, gpu_storage->lens1,
				gpu_storage->lens2, gpu_storage->offsets1, gpu_storage->offsets2, gpu_storage->aln_score,
				gpu_storage->batch1_end, gpu_storage->batch2_end, gpu_storage->batch1_start,
				gpu_storage->batch2_start, actual_n_alns);
	} else {
		gasal_local_kernel<<<N_BLOCKS, BLOCKDIM, 0, gpu_storage->str>>>(gpu_storage->packed1_4bit, gpu_storage->packed2_4bit, gpu_storage->lens1,
				gpu_storage->lens2, gpu_storage->offsets1, gpu_storage->offsets2, gpu_storage->aln_score,
				gpu_storage->batch1_end, gpu_storage->batch2_end, actual_n_alns);
	}

    cudaError_t aln_kernel_err = cudaGetLastError();
    if ( cudaSuccess != aln_kernel_err )
    {
    	fprintf(stderr, "Cuda error:%d(%s) at line no. %d in file %s\n", aln_kernel_err, cudaGetErrorString(aln_kernel_err), __LINE__, __FILE__);
    	exit(EXIT_FAILURE);
    }

    gpu_storage->n_alns = actual_n_alns;
    gpu_storage->is_free = 0;

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

	CHECKCUDAERROR(cudaStreamDestroy(gpu_storage->str));
	if (gpu_storage != NULL) free(gpu_storage);
	else {
		fprintf(stderr, "Pointer to gasal_gpu_storage_t can't be NULL\n");
		exit(EXIT_FAILURE);
	}


	return 0;
}

int gasal_is_aln_async_done_new(gasal_gpu_storage_t *gpu_storage) {

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
//	if(gpu_storage->is_gpu_mem_alloc_contig == 1 && gpu_storage->is_host_mem_alloc_contig == 1) CHECKCUDAERROR(cudaMemcpy(gpu_storage->host_aln_score, gpu_storage->aln_score, 5 * gpu_storage->n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost));
//	else {
//		if (gpu_storage->host_aln_score != NULL && gpu_storage->aln_score != NULL) CHECKCUDAERROR(cudaMemcpy(gpu_storage->host_aln_score, gpu_storage->aln_score, gpu_storage->n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost));
//		if (gpu_storage->host_batch1_start != NULL && gpu_storage->batch1_start != NULL) CHECKCUDAERROR(cudaMemcpy(gpu_storage->host_batch1_start, gpu_storage->batch1_start, gpu_storage->n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost));
//		if (gpu_storage->host_batch2_start != NULL && gpu_storage->batch2_start != NULL) CHECKCUDAERROR(cudaMemcpy(gpu_storage->host_batch2_start, gpu_storage->batch2_start, gpu_storage->n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost));
//		if (gpu_storage->host_batch1_end != NULL && gpu_storage->batch1_end != NULL) CHECKCUDAERROR(cudaMemcpy(gpu_storage->host_batch1_end, gpu_storage->batch1_end, gpu_storage->n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost));
//		if (gpu_storage->host_batch2_end != NULL && gpu_storage->batch2_end != NULL) CHECKCUDAERROR(cudaMemcpy(gpu_storage->host_batch2_end, gpu_storage->batch2_end, gpu_storage->n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost));
//	}



	return 0;
}

int gasal_is_aln_async_done_new2(gasal_gpu_storage_t *gpu_storage) {

	cudaError_t err;
	err = cudaStreamQuery(gpu_storage->str);
	if (err != cudaSuccess ) {
		if (err == cudaErrorNotReady) return -1;
		else{
			fprintf(stderr, "Cuda error:%d(%s) at line no. %d in file %s\n", err, cudaGetErrorString(err), __LINE__, __FILE__);
			exit(EXIT_FAILURE);
		}
	}
	if (gpu_storage->host_results != NULL && gpu_storage->aln_score != NULL) CHECKCUDAERROR(cudaMemcpy(gpu_storage->host_results, gpu_storage->aln_score, 5 * gpu_storage->n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost));
	if (gpu_storage->unpacked1 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->unpacked1));
	if (gpu_storage->str != NULL)CHECKCUDAERROR(cudaStreamDestroy(gpu_storage->str));

	return 0;
}


//void gasal_get_aln_async_results(int32_t *host_aln_score, int32_t *host_batch1_start, int32_t *host_batch2_start, int32_t *host_batch1_end, int32_t *host_batch2_end, gasal_gpu_storage_t *gpu_storage) {
//
//	cudaError_t err;
//
//	if (host_aln_score != NULL && gpu_storage->aln_score != NULL) CHECKCUDAERROR(cudaMemcpy(host_aln_score, gpu_storage->aln_score, gpu_storage->n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost));
//	else {
//		fprintf(stderr, "The *host_aln_score input can't be NULL\n");
//		exit(EXIT_FAILURE);
//	}
//	if (host_batch1_start != NULL && gpu_storage->batch1_start != NULL) CHECKCUDAERROR(cudaMemcpy(host_batch1_start, gpu_storage->batch1_start, gpu_storage->n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost));
//	if (host_batch2_start != NULL && gpu_storage->batch2_start != NULL) CHECKCUDAERROR(cudaMemcpy(host_batch2_start, gpu_storage->batch2_start, gpu_storage->n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost));
//	if (host_batch1_end != NULL && gpu_storage->batch1_end != NULL) CHECKCUDAERROR(cudaMemcpy(host_batch1_end, gpu_storage->batch1_end, gpu_storage->n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost));
//	if (host_batch2_end != NULL && gpu_storage->batch2_end != NULL) CHECKCUDAERROR(cudaMemcpy(host_batch2_end, gpu_storage->batch2_end, gpu_storage->n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost));
//
//
//
//}

//void gasal_free_gpu_storage( gasal_gpu_storage_t *gpu_storage) {
//
//	cudaError_t err;
//
//	if (gpu_storage->unpacked1 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->unpacked1));
//	if (gpu_storage->unpacked2 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->unpacked2));
//	if (gpu_storage->packed1_4bit != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->packed1_4bit));
//	if (gpu_storage->packed2_4bit != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->packed2_4bit));
//	if (gpu_storage->offsets1 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->offsets1));
//	if (gpu_storage->offsets2 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->offsets2));
//	if (gpu_storage->lens1 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->lens1));
//	if (gpu_storage->lens2 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->lens2));
//	if (gpu_storage->aln_score != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->aln_score));
//	if (gpu_storage->batch1_start != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->batch1_start));
//	if (gpu_storage->batch2_start != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->batch2_start));
//	if (gpu_storage->batch1_end != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->batch1_end));
//	if (gpu_storage->batch2_end != NULL) CHECKCUDAERROR(cudaFree(gpu_storage->batch2_end));
//
//
//}
//
//
//void gasal_stream_destroy( gasal_gpu_storage_t *gpu_storage) {
//
//	cudaError_t err;
//
//
//	CHECKCUDAERROR(cudaStreamDestroy(gpu_storage->str));
//	if (gpu_storage != NULL) free(gpu_storage);
//	else {
//		fprintf(stderr, "Pointer to gasal_gpu_storage_t can't be NULL\n");
//		exit(EXIT_FAILURE);
//	}
//
//}
void gasal_aln_imp_mem_alloc(gasal_gpu_storage_t *gpu_storage, int algo, int start) {

	cudaError_t err;
	if (gpu_storage->max_batch1_bytes % 8) {
		fprintf(stderr, "max_batch1_bytes should be multiple of 8\n");
		exit(EXIT_FAILURE);
	}
	if (gpu_storage->max_batch2_bytes % 8) {
		fprintf(stderr, "max_batch2_bytes should be multiple of 8\n");
		exit(EXIT_FAILURE);
	}
	CHECKCUDAERROR(cudaMalloc(&(gpu_storage->unpacked1), gpu_storage->max_batch1_bytes * sizeof(uint8_t)));
	CHECKCUDAERROR(cudaMalloc(&(gpu_storage->unpacked2), gpu_storage->max_batch2_bytes * sizeof(uint8_t)));

	CHECKCUDAERROR(cudaMalloc(&(gpu_storage->packed1_4bit), (gpu_storage->max_batch1_bytes/8) * sizeof(uint32_t)));
	CHECKCUDAERROR(cudaMalloc(&(gpu_storage->packed2_4bit), (gpu_storage->max_batch2_bytes/8) * sizeof(uint32_t)));

	CHECKCUDAERROR(cudaMalloc(&(gpu_storage->lens1), gpu_storage->max_n_alns * sizeof(uint32_t)));
	CHECKCUDAERROR(cudaMalloc(&(gpu_storage->lens2), gpu_storage->max_n_alns * sizeof(uint32_t)));
	CHECKCUDAERROR(cudaMalloc(&(gpu_storage->offsets1), gpu_storage->max_n_alns * sizeof(uint32_t)));
	CHECKCUDAERROR(cudaMalloc(&(gpu_storage->offsets2), gpu_storage->max_n_alns * sizeof(uint32_t)));

	CHECKCUDAERROR(cudaMalloc(&(gpu_storage->aln_score),gpu_storage->max_n_alns * sizeof(int32_t)));
	if (algo == GLOBAL) {
		gpu_storage->batch1_start = NULL;
		gpu_storage->batch1_end = NULL;
		gpu_storage->batch2_start = NULL;
		gpu_storage->batch2_end = NULL;
	} else {
		CHECKCUDAERROR(
				cudaMalloc(&(gpu_storage->batch2_end),
						gpu_storage->max_n_alns * sizeof(uint32_t)));
		if (start == WITH_START) {
			CHECKCUDAERROR(
					cudaMalloc(&(gpu_storage->batch2_start),
							gpu_storage->max_n_alns * sizeof(uint32_t)));
		} else
			gpu_storage->batch2_start = NULL;
		if (algo == LOCAL) {
			CHECKCUDAERROR(
					cudaMalloc(&(gpu_storage->batch1_end),
							gpu_storage->max_n_alns * sizeof(uint32_t)));
			if (start == WITH_START) {
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage->batch1_start),
								gpu_storage->max_n_alns * sizeof(uint32_t)));
			} else
				gpu_storage->batch1_start = NULL;
		} else {
			gpu_storage->batch1_start = NULL;
			gpu_storage->batch1_end = NULL;
		}
	}


}



void gasal_host_mem_alloc(gasal_gpu_storage_t *gpu_storage, int max_batch1_bytes, int max_batch2_bytes, int max_n_alns, int algo, int start) {

	cudaError_t err;
	if (max_batch1_bytes % 8) {
		fprintf(stderr, "max_batch1_bytes should be multiple of 8\n");
		exit(EXIT_FAILURE);
	}
	if (max_batch2_bytes % 8) {
		fprintf(stderr, "max_batch2_bytes should be multiple of 8\n");
		exit(EXIT_FAILURE);
	}
	//gasal_gpu_storage_t *gpu_storage = (gasal_gpu_storage_t*)calloc(1, sizeof(gasal_gpu_storage_t));

	gpu_storage->host_max_batch1_bytes = HOST_MALLOC_SAFETY_FACTOR * max_batch1_bytes;
	gpu_storage->host_max_batch2_bytes = HOST_MALLOC_SAFETY_FACTOR * max_batch2_bytes;
	gpu_storage->host_max_n_alns = HOST_MALLOC_SAFETY_FACTOR * max_n_alns;

	uint64_t total_host_malloc_size;


	if (algo == GLOBAL) {
		total_host_malloc_size = gpu_storage->host_max_batch1_bytes + gpu_storage->host_max_batch2_bytes + (5 * gpu_storage->host_max_n_alns * sizeof(uint32_t));
		CHECKCUDAERROR( cudaMallocHost(&(gpu_storage->host_unpacked1),
				total_host_malloc_size));
		gpu_storage->host_unpacked2 = &(gpu_storage->host_unpacked1[gpu_storage->host_max_batch1_bytes]);
		gpu_storage->host_offsets1 = (uint32_t*)&(gpu_storage->host_unpacked2[gpu_storage->host_max_batch2_bytes]);
		gpu_storage->host_offsets2 = &(gpu_storage->host_offsets1[gpu_storage->host_max_n_alns]);
		gpu_storage->host_lens1 = &(gpu_storage->host_offsets2[gpu_storage->host_max_n_alns]);
		gpu_storage->host_lens2 = &(gpu_storage->host_lens1[gpu_storage->host_max_n_alns]);
		gpu_storage->host_aln_score = (int32_t*)&(gpu_storage->host_lens2[gpu_storage->host_max_n_alns]);
		gpu_storage->host_batch1_start = NULL;
		gpu_storage->host_batch2_start = NULL;
		gpu_storage->host_batch1_end = NULL;
		gpu_storage->host_batch2_end = NULL;



	} else if (algo == SEMI_GLOBAL){
		if (start == WITH_START) {
			total_host_malloc_size = gpu_storage->host_max_batch1_bytes + gpu_storage->host_max_batch2_bytes + (7 * gpu_storage->host_max_n_alns * sizeof(uint32_t));
			CHECKCUDAERROR( cudaMallocHost(&(gpu_storage->host_unpacked1),
					total_host_malloc_size));
			gpu_storage->host_unpacked2 = &(gpu_storage->host_unpacked1[gpu_storage->host_max_batch1_bytes]);
			gpu_storage->host_offsets1 = (uint32_t*)&(gpu_storage->host_unpacked2[gpu_storage->host_max_batch2_bytes]);
			gpu_storage->host_offsets2 = &(gpu_storage->host_offsets1[gpu_storage->host_max_n_alns]);
			gpu_storage->host_lens1 = &(gpu_storage->host_offsets2[gpu_storage->host_max_n_alns]);
			gpu_storage->host_lens2 = &(gpu_storage->host_lens1[gpu_storage->host_max_n_alns]);
			gpu_storage->host_aln_score = (int32_t*)&(gpu_storage->host_lens2[gpu_storage->host_max_n_alns]);
			gpu_storage->host_batch1_start = NULL;
			gpu_storage->host_batch2_start = &(gpu_storage->host_aln_score[gpu_storage->host_max_n_alns]);
			gpu_storage->host_batch1_end = NULL;
			gpu_storage->host_batch2_end = &(gpu_storage->host_batch2_start[gpu_storage->host_max_n_alns]);
		} else {
			total_host_malloc_size = gpu_storage->host_max_batch1_bytes + gpu_storage->host_max_batch2_bytes + (6 * gpu_storage->host_max_n_alns * sizeof(uint32_t));
			CHECKCUDAERROR( cudaMallocHost(&(gpu_storage->host_unpacked1),
					total_host_malloc_size));
			gpu_storage->host_unpacked2 = &(gpu_storage->host_unpacked1[gpu_storage->host_max_batch1_bytes]);
			gpu_storage->host_offsets1 = (uint32_t*)&(gpu_storage->host_unpacked2[gpu_storage->host_max_batch2_bytes]);
			gpu_storage->host_offsets2 = &(gpu_storage->host_offsets1[gpu_storage->host_max_n_alns]);
			gpu_storage->host_lens1 = &(gpu_storage->host_offsets2[gpu_storage->host_max_n_alns]);
			gpu_storage->host_lens2 = &(gpu_storage->host_lens1[gpu_storage->host_max_n_alns]);
			gpu_storage->host_aln_score = (int32_t*)&(gpu_storage->host_lens2[gpu_storage->host_max_n_alns]);
			gpu_storage->host_batch1_start = NULL;
			gpu_storage->host_batch2_start = NULL;
			gpu_storage->host_batch1_end = NULL;
			gpu_storage->host_batch2_end = &(gpu_storage->host_aln_score[gpu_storage->host_max_n_alns]);
		}
	} else {
		if (start == WITH_START) {
			total_host_malloc_size = gpu_storage->host_max_batch1_bytes + gpu_storage->host_max_batch2_bytes + (9 * gpu_storage->host_max_n_alns * sizeof(uint32_t));
			CHECKCUDAERROR( cudaMallocHost(&(gpu_storage->host_unpacked1),
					total_host_malloc_size));
			gpu_storage->host_unpacked2 = &(gpu_storage->host_unpacked1[gpu_storage->host_max_batch1_bytes]);
			gpu_storage->host_offsets1 = (uint32_t*)&(gpu_storage->host_unpacked2[gpu_storage->host_max_batch2_bytes]);
			gpu_storage->host_offsets2 = (uint32_t*)&(gpu_storage->host_offsets1[gpu_storage->host_max_n_alns]);
			gpu_storage->host_lens1 = (uint32_t*)&(gpu_storage->host_offsets2[gpu_storage->host_max_n_alns]);
			gpu_storage->host_lens2 = (uint32_t*)&(gpu_storage->host_lens1[gpu_storage->host_max_n_alns]);
			gpu_storage->host_aln_score = (int32_t*)&(gpu_storage->host_lens2[gpu_storage->host_max_n_alns]);
			gpu_storage->host_batch1_start = (int32_t*)&(gpu_storage->host_aln_score[gpu_storage->host_max_n_alns]);
			gpu_storage->host_batch2_start = (int32_t*)&(gpu_storage->host_batch1_start[gpu_storage->host_max_n_alns]);
			gpu_storage->host_batch1_end = (int32_t*)&(gpu_storage->host_batch2_start[gpu_storage->host_max_n_alns]);
			gpu_storage->host_batch2_end =  (int32_t*)&(gpu_storage->host_batch1_end[gpu_storage->host_max_n_alns]);
		} else {
			total_host_malloc_size = gpu_storage->host_max_batch1_bytes + gpu_storage->host_max_batch2_bytes + (7 * gpu_storage->host_max_n_alns * sizeof(uint32_t));
			CHECKCUDAERROR( cudaMallocHost(&(gpu_storage->host_unpacked1),
					total_host_malloc_size));
			gpu_storage->host_unpacked2 = &(gpu_storage->host_unpacked1[gpu_storage->host_max_batch1_bytes]);
			gpu_storage->host_offsets1 = (uint32_t*)&(gpu_storage->host_unpacked2[gpu_storage->host_max_batch2_bytes]);
			gpu_storage->host_offsets2 = &(gpu_storage->host_offsets1[gpu_storage->host_max_n_alns]);
			gpu_storage->host_lens1 = &(gpu_storage->host_offsets2[gpu_storage->host_max_n_alns]);
			gpu_storage->host_lens2 = &(gpu_storage->host_lens1[gpu_storage->host_max_n_alns]);
			gpu_storage->host_aln_score = (int32_t*)&(gpu_storage->host_lens2[gpu_storage->host_max_n_alns]);
			gpu_storage->host_batch1_start = NULL;
			gpu_storage->host_batch2_start = NULL;
			gpu_storage->host_batch1_end = &(gpu_storage->host_aln_score[gpu_storage->host_max_n_alns]);
			gpu_storage->host_batch2_end = &(gpu_storage->host_batch1_end[gpu_storage->host_max_n_alns]);
		}

	}


}

void gasal_gpu_mem_alloc(gasal_gpu_storage_t *gpu_storage, const uint32_t actual_batch1_bytes, const uint32_t actual_batch2_bytes, const uint32_t actual_n_alns, int algo, int start) {

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

	uint64_t total_gpu_malloc_size;



	if (algo == GLOBAL) {
		total_gpu_malloc_size = actual_batch1_bytes + actual_batch2_bytes  + (actual_batch1_bytes/2) + (actual_batch2_bytes/2) + (5 * actual_n_alns * sizeof(uint32_t)) + (8*128);
		CHECKCUDAERROR( cudaMalloc(&(gpu_storage->unpacked1),
				total_gpu_malloc_size));
		gpu_storage->unpacked2 = &(gpu_storage->unpacked1[actual_batch1_bytes]);
		gpu_storage->packed1_4bit = (uint32_t*)&(gpu_storage->unpacked2[actual_batch2_bytes]);
		gpu_storage->packed2_4bit = (uint32_t*)&(gpu_storage->packed1_4bit[actual_batch1_bytes/8]);
		gpu_storage->offsets1 = (uint32_t*)&(gpu_storage->packed2_4bit[actual_batch2_bytes/8]);
		gpu_storage->offsets2 = &(gpu_storage->offsets1[actual_n_alns]);
		gpu_storage->lens1 = &(gpu_storage->offsets2[actual_n_alns]);
		gpu_storage->lens2 = &(gpu_storage->lens1[actual_n_alns]);
		gpu_storage->aln_score = (int32_t*)&(gpu_storage->lens2[actual_n_alns]);
		gpu_storage->batch1_start = NULL;
		gpu_storage->batch2_start = NULL;
		gpu_storage->batch1_end = NULL;
		gpu_storage->batch2_end = NULL;



	} else if (algo == SEMI_GLOBAL){
		if (start == WITH_START) {
			total_gpu_malloc_size = actual_batch1_bytes + actual_batch2_bytes  + (actual_batch1_bytes/2) + (actual_batch2_bytes/2) + (7 * actual_n_alns * sizeof(uint32_t)) + (10*128);
			CHECKCUDAERROR( cudaMalloc(&(gpu_storage->unpacked1),
					total_gpu_malloc_size));
			gpu_storage->unpacked2 = &(gpu_storage->unpacked1[actual_batch1_bytes]);
			gpu_storage->packed1_4bit = (uint32_t*)&(gpu_storage->unpacked2[actual_batch2_bytes]);
			gpu_storage->packed2_4bit = (uint32_t*)&(gpu_storage->packed1_4bit[actual_batch1_bytes/8]);
			gpu_storage->offsets1 = (uint32_t*)&(gpu_storage->packed2_4bit[actual_batch2_bytes/8]);
			gpu_storage->offsets2 = &(gpu_storage->offsets1[actual_n_alns]);
			gpu_storage->lens1 = &(gpu_storage->offsets2[actual_n_alns]);
			gpu_storage->lens2 = &(gpu_storage->lens1[actual_n_alns]);
			gpu_storage->aln_score = (int32_t*)&(gpu_storage->lens2[actual_n_alns]);
			gpu_storage->batch1_start = NULL;
			gpu_storage->batch2_start = &(gpu_storage->aln_score[actual_n_alns]);
			gpu_storage->batch1_end = NULL;
			gpu_storage->batch2_end = &(gpu_storage->batch2_start[actual_n_alns]);

		} else {
			total_gpu_malloc_size = actual_batch1_bytes + actual_batch2_bytes  + (actual_batch1_bytes/2) + (actual_batch2_bytes/2) + (6 * actual_n_alns * sizeof(uint32_t)) +  (9*128);
			CHECKCUDAERROR( cudaMalloc(&(gpu_storage->unpacked1),
					total_gpu_malloc_size));
			gpu_storage->unpacked2 = &(gpu_storage->unpacked1[actual_batch1_bytes]);
			gpu_storage->packed1_4bit = (uint32_t*)&(gpu_storage->unpacked2[actual_batch2_bytes]);
			gpu_storage->packed2_4bit = (uint32_t*)&(gpu_storage->packed1_4bit[actual_batch1_bytes/8]);
			gpu_storage->offsets1 = (uint32_t*)&(gpu_storage->packed2_4bit[actual_batch2_bytes/8]);
			gpu_storage->offsets2 = &(gpu_storage->offsets1[actual_n_alns]);
			gpu_storage->lens1 = &(gpu_storage->offsets2[actual_n_alns]);
			gpu_storage->lens2 = &(gpu_storage->lens1[actual_n_alns]);
			gpu_storage->aln_score = (int32_t*)&(gpu_storage->lens2[actual_n_alns]);
			gpu_storage->batch1_start = NULL;
			gpu_storage->batch2_start =NULL;
			gpu_storage->batch1_end = NULL;
			gpu_storage->batch2_end = &(gpu_storage->aln_score[actual_n_alns]);

		}
	} else {
		if (start == WITH_START) {
			total_gpu_malloc_size = actual_batch1_bytes + actual_batch2_bytes  + (actual_batch1_bytes/2) + (actual_batch2_bytes/2) + (9 * actual_n_alns * sizeof(uint32_t));
			CHECKCUDAERROR( cudaMalloc(&(gpu_storage->unpacked1),
					total_gpu_malloc_size));
			gpu_storage->unpacked2 = &(gpu_storage->unpacked1[actual_batch1_bytes]);
			gpu_storage->packed1_4bit = (uint32_t*)&(gpu_storage->unpacked2[actual_batch2_bytes]);
			gpu_storage->packed2_4bit = (uint32_t*)&(gpu_storage->packed1_4bit[actual_batch1_bytes/8]);
			gpu_storage->offsets1 = (uint32_t*)&(gpu_storage->packed2_4bit[actual_batch2_bytes/8]);
			gpu_storage->offsets2 = (uint32_t*)&(gpu_storage->offsets1[actual_n_alns]);
			gpu_storage->lens1 = (uint32_t*)&(gpu_storage->offsets2[actual_n_alns]);
			gpu_storage->lens2 = (uint32_t*)&(gpu_storage->lens1[actual_n_alns]);
			gpu_storage->aln_score = (int32_t*)&(gpu_storage->lens2[actual_n_alns]);
			gpu_storage->batch1_start = (int32_t*)&(gpu_storage->aln_score[actual_n_alns]);
			gpu_storage->batch2_start =(int32_t*)&(gpu_storage->batch1_start[actual_n_alns]);
			gpu_storage->batch1_end = (int32_t*)&(gpu_storage->batch2_start[actual_n_alns]);
			gpu_storage->batch2_end = (int32_t*)&(gpu_storage->batch1_end[actual_n_alns]);
		} else {
			total_gpu_malloc_size = actual_batch1_bytes + actual_batch2_bytes  + (actual_batch1_bytes/2) + (actual_batch2_bytes/2) + (7 * actual_n_alns * sizeof(uint32_t)) + (10*128);
			CHECKCUDAERROR( cudaMalloc(&(gpu_storage->unpacked1),
					total_gpu_malloc_size));
			//int next_addr = (actual_batch1_bytes % 128) ? actual_batch1_bytes;
			gpu_storage->unpacked2 = &(gpu_storage->unpacked1[actual_batch1_bytes]);
			gpu_storage->packed1_4bit = (uint32_t*)&(gpu_storage->unpacked2[actual_batch2_bytes]);
			gpu_storage->packed2_4bit = (uint32_t*)&(gpu_storage->packed1_4bit[actual_batch1_bytes/8]);
			gpu_storage->offsets1 = (uint32_t*)&(gpu_storage->packed2_4bit[actual_batch2_bytes/8]);
			gpu_storage->offsets2 = &(gpu_storage->offsets1[actual_n_alns]);
			gpu_storage->lens1 = &(gpu_storage->offsets2[actual_n_alns]);
			gpu_storage->lens2 = &(gpu_storage->lens1[actual_n_alns]);
			gpu_storage->aln_score = (int32_t*)&(gpu_storage->lens2[actual_n_alns]);
			gpu_storage->batch1_start = NULL;
			gpu_storage->batch2_start =NULL;
			gpu_storage->batch1_end = &(gpu_storage->aln_score[actual_n_alns]);
			gpu_storage->batch2_end = &(gpu_storage->batch1_end[actual_n_alns]);
		}

	}


}

//void gasal_gpu_mem_alloc_contig(gasal_gpu_storage_t *gpu_storage, const uint32_t max_batch1_bytes, const uint32_t max_batch2_bytes, const uint32_t max_n_alns, int algo, int start) {
//
//	cudaError_t err;
//	if (max_n_alns <= 0) {
//		fprintf(stderr, "Must perform at least 1 alignment (n_alns > 0)\n");
//		exit(EXIT_FAILURE);
//	}
//	if (max_batch1_bytes <= 0) {
//		fprintf(stderr, "Number of batch1_bytes should be greater than 0\n");
//		exit(EXIT_FAILURE);
//	}
//	if (max_batch2_bytes <= 0) {
//		fprintf(stderr, "Number of batch2_bytes should be greater than 0\n");
//		exit(EXIT_FAILURE);
//	}
//
//	if (max_batch1_bytes % 8) {
//		fprintf(stderr, "Number of batch1_bytes should be multiple of 8\n");
//		exit(EXIT_FAILURE);
//	}
//	if (max_batch2_bytes % 8) {
//		fprintf(stderr, "Number of batch2_bytes should be multiple of 8\n");
//		exit(EXIT_FAILURE);
//
//	}
//
//
//
//	if (algo == GLOBAL) {
//		gpu_storage->max_gpu_malloc_size = max_batch1_bytes + max_batch2_bytes  + (max_batch1_bytes/2) + (max_batch2_bytes/2) + (5 * max_n_alns * sizeof(uint32_t)) + (8*128);
//		CHECKCUDAERROR( cudaMalloc(&(gpu_storage->unpacked1),
//				gpu_storage->max_gpu_malloc_size));
//
//
//
//	} else if (algo == SEMI_GLOBAL){
//		if (start == WITH_START) {
//			gpu_storage->max_gpu_malloc_size = max_batch1_bytes + max_batch2_bytes  + (max_batch1_bytes/2) + (max_batch2_bytes/2) + (7 * max_n_alns * sizeof(uint32_t)) + (10*128);
//			CHECKCUDAERROR( cudaMalloc(&(gpu_storage->unpacked1),
//					gpu_storage->max_gpu_malloc_size));
//		} else {
//			gpu_storage->max_gpu_malloc_size = max_batch1_bytes + max_batch2_bytes  + (max_batch1_bytes/2) + (max_batch2_bytes/2) + (6 * max_n_alns * sizeof(uint32_t)) +  (9*128);
//			CHECKCUDAERROR( cudaMalloc(&(gpu_storage->unpacked1),
//					gpu_storage->max_gpu_malloc_size));
//		}
//	} else {
//		if (start == WITH_START) {
//			gpu_storage->max_gpu_malloc_size = max_batch1_bytes + max_batch2_bytes  + (max_batch1_bytes/2) + (max_batch2_bytes/2) + (9 * max_n_alns * sizeof(uint32_t));
//			CHECKCUDAERROR( cudaMalloc(&(gpu_storage->unpacked1),
//					gpu_storage->max_gpu_malloc_size));
//		} else {
//			gpu_storage->max_gpu_malloc_size = max_batch1_bytes + max_batch2_bytes  + (max_batch1_bytes/2) + (max_batch2_bytes/2) + (7 * max_n_alns * sizeof(uint32_t)) + (10*128);
//			CHECKCUDAERROR( cudaMalloc(&(gpu_storage->unpacked1),
//					gpu_storage->max_gpu_malloc_size));
//		}
//
//	}
//
//	gpu_storage->is_contig = 1;
//
//
//}



void gasal_init_streams(gasal_gpu_storage_v *gpu_storage_vec, int max_batch1_bytes, int max_batch2_bytes, int max_n_alns, int algo, int start) {

	cudaError_t err;
	if (max_batch1_bytes % 8) {
		fprintf(stderr, "max_batch1_bytes should be multiple of 8\n");
		exit(EXIT_FAILURE);
	}
	if (max_batch2_bytes % 8) {
		fprintf(stderr, "max_batch2_bytes should be multiple of 8\n");
		exit(EXIT_FAILURE);
	}
	int i;
	for (i = 0; i < gpu_storage_vec->n; i++) {


		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].unpacked1), max_batch1_bytes * sizeof(uint8_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].unpacked2), max_batch2_bytes * sizeof(uint8_t)));

		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].packed1_4bit), (max_batch1_bytes/8) * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].packed2_4bit), (max_batch2_bytes/8) * sizeof(uint32_t)));

		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].lens1), max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].lens2), max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].offsets1), max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].offsets2), max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].aln_score), max_n_alns * sizeof(int32_t)));
		if (algo == GLOBAL) {
			gpu_storage_vec->a[i].batch1_start = NULL;
			gpu_storage_vec->a[i].batch2_start = NULL;
			gpu_storage_vec->a[i].batch1_end = NULL;
			gpu_storage_vec->a[i].batch2_end = NULL;
		} else if (algo == SEMI_GLOBAL) {
			gpu_storage_vec->a[i].batch1_start = NULL;
			gpu_storage_vec->a[i].batch1_end = NULL;
			if (start == WITH_START) {
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage_vec->a[i].batch2_start),
								max_n_alns * sizeof(uint32_t)));
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage_vec->a[i].batch2_end),
								max_n_alns * sizeof(uint32_t)));
			} else {
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage_vec->a[i].batch2_end),
								max_n_alns * sizeof(uint32_t)));
				gpu_storage_vec->a[i].batch2_start = NULL;
			}
		} else {
			if (start == WITH_START) {
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage_vec->a[i].batch1_start),
								max_n_alns * sizeof(uint32_t)));
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage_vec->a[i].batch2_start),
								max_n_alns * sizeof(uint32_t)));
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage_vec->a[i].batch1_end),
								max_n_alns * sizeof(uint32_t)));
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage_vec->a[i].batch2_end),
								max_n_alns * sizeof(uint32_t)));
			} else {
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage_vec->a[i].batch1_end),
								max_n_alns * sizeof(uint32_t)));
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage_vec->a[i].batch2_end),
								max_n_alns * sizeof(uint32_t)));
				gpu_storage_vec->a[i].batch1_start = NULL;
				gpu_storage_vec->a[i].batch2_start = NULL;
			}
		}


		CHECKCUDAERROR(cudaStreamCreate(&(gpu_storage_vec->a[i].str)));
		gpu_storage_vec->a[i].is_free = 1;
		gpu_storage_vec->a[i].max_batch1_bytes = max_batch1_bytes;
		gpu_storage_vec->a[i].max_batch2_bytes = max_batch2_bytes;
		gpu_storage_vec->a[i].max_n_alns = max_n_alns;
		gpu_storage_vec->a[i].is_gpu_mem_alloc_contig = 0;

	}


}

void gasal_init_streams_with_host_malloc(gasal_gpu_storage_v *gpu_storage_vec, int host_max_batch1_bytes,  int max_batch1_bytes,  int host_max_batch2_bytes, int max_batch2_bytes, int host_max_n_alns, int max_n_alns, int algo, int start) {

	cudaError_t err;
//	if (max_batch1_bytes % 8) {
//		fprintf(stderr, "max_batch1_bytes should be multiple of 8\n");
//		exit(EXIT_FAILURE);
//	}
//	if (max_batch2_bytes % 8) {
//		fprintf(stderr, "max_batch2_bytes should be multiple of 8\n");
//		exit(EXIT_FAILURE);
//	}
	int i;
	for (i = 0; i < gpu_storage_vec->n; i++) {


		CHECKCUDAERROR(cudaMallocHost(&(gpu_storage_vec->a[i].host_unpacked1), host_max_batch1_bytes * sizeof(uint8_t)));
		CHECKCUDAERROR(cudaMallocHost(&(gpu_storage_vec->a[i].host_unpacked2), host_max_batch2_bytes * sizeof(uint8_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].unpacked1), max_batch1_bytes * sizeof(uint8_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].unpacked2), max_batch2_bytes * sizeof(uint8_t)));

		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].packed1_4bit), (max_batch1_bytes/8) * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].packed2_4bit), (max_batch2_bytes/8) * sizeof(uint32_t)));


		CHECKCUDAERROR(cudaMallocHost(&(gpu_storage_vec->a[i].host_lens1), host_max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMallocHost(&(gpu_storage_vec->a[i].host_lens2), host_max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMallocHost(&(gpu_storage_vec->a[i].host_offsets1), host_max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMallocHost(&(gpu_storage_vec->a[i].host_offsets2), host_max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMallocHost(&(gpu_storage_vec->a[i].host_aln_score), host_max_n_alns * sizeof(int32_t)));

		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].lens1), max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].lens2), max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].offsets1), max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].offsets2), max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].aln_score), max_n_alns * sizeof(int32_t)));
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
								max_n_alns * sizeof(uint32_t)));
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage_vec->a[i].batch2_end),
								max_n_alns * sizeof(uint32_t)));
			} else {
				CHECKCUDAERROR(cudaMallocHost(&(gpu_storage_vec->a[i].host_batch2_end),host_max_n_alns * sizeof(uint32_t)));
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage_vec->a[i].batch2_end),
								max_n_alns * sizeof(uint32_t)));
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
								max_n_alns * sizeof(uint32_t)));
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage_vec->a[i].batch2_start),
								max_n_alns * sizeof(uint32_t)));
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage_vec->a[i].batch1_end),
								max_n_alns * sizeof(uint32_t)));
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage_vec->a[i].batch2_end),
								max_n_alns * sizeof(uint32_t)));
			} else {
				CHECKCUDAERROR(cudaMallocHost(&(gpu_storage_vec->a[i].host_batch1_end),host_max_n_alns * sizeof(uint32_t)));
				CHECKCUDAERROR(cudaMallocHost(&(gpu_storage_vec->a[i].host_batch2_end),host_max_n_alns * sizeof(uint32_t)));
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage_vec->a[i].batch1_end),
								max_n_alns * sizeof(uint32_t)));
				CHECKCUDAERROR(
						cudaMalloc(&(gpu_storage_vec->a[i].batch2_end),
								max_n_alns * sizeof(uint32_t)));
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
		gpu_storage_vec->a[i].max_batch1_bytes = max_batch1_bytes;
		gpu_storage_vec->a[i].max_batch2_bytes = max_batch2_bytes;
		gpu_storage_vec->a[i].max_n_alns = max_n_alns;
		gpu_storage_vec->a[i].is_host_mem_alloc_contig = 0;
		gpu_storage_vec->a[i].is_gpu_mem_alloc_contig = 0;

	}


}

void gasal_init_streams_new(gasal_gpu_storage_v *gpu_storage_vec, int max_batch1_bytes, int max_batch2_bytes, int max_n_alns, int algo, int start) {

	cudaError_t err;
	if (max_batch1_bytes % 8) {
		fprintf(stderr, "max_batch1_bytes should be multiple of 8\n");
		exit(EXIT_FAILURE);
	}
	if (max_batch2_bytes % 8) {
		fprintf(stderr, "max_batch2_bytes should be multiple of 8\n");
		exit(EXIT_FAILURE);
	}
	int i;
	for (i = 0; i < gpu_storage_vec->n; i++) {
		gpu_storage_vec->a[i].is_gpu_mem_alloc = 0;
		gasal_gpu_mem_alloc_contig(&(gpu_storage_vec->a[i]), max_batch1_bytes, max_batch2_bytes, max_n_alns, algo, start);

		CHECKCUDAERROR(cudaStreamCreate(&(gpu_storage_vec->a[i].str)));
		gpu_storage_vec->a[i].is_free = 1;
		gpu_storage_vec->a[i].max_batch1_bytes = max_batch1_bytes;
		gpu_storage_vec->a[i].max_batch2_bytes = max_batch2_bytes;
		gpu_storage_vec->a[i].max_n_alns = max_n_alns;


	}


}
void gasal_aln_imp_mem_free(gasal_gpu_storage_t *gpu_storage) {

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
		if (gpu_storage_vec->a[i].is_gpu_mem_alloc_contig) {
			if (gpu_storage_vec->a[i].unpacked1 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage_vec->a[i].unpacked1));
		}
		else {
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
		}
		if (gpu_storage_vec->a[i].str != NULL)CHECKCUDAERROR(cudaStreamDestroy(gpu_storage_vec->a[i].str));
	}



}


void gasal_destroy_streams_with_host_malloc(gasal_gpu_storage_v *gpu_storage_vec) {

	cudaError_t err;

	int i;
	for (i = 0; i < gpu_storage_vec->n; i ++) {
		if (gpu_storage_vec->a[i].is_host_mem_alloc_contig) {
			if (gpu_storage_vec->a[i].host_unpacked1 != NULL) CHECKCUDAERROR(cudaFreeHost(gpu_storage_vec->a[i].host_unpacked1));
		}
		else {
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
		}
		if (gpu_storage_vec->a[i].is_gpu_mem_alloc_contig) {
			if (gpu_storage_vec->a[i].unpacked1 != NULL) CHECKCUDAERROR(cudaFree(gpu_storage_vec->a[i].unpacked1));
		}
		else {
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
		}
		if (gpu_storage_vec->a[i].str != NULL)CHECKCUDAERROR(cudaStreamDestroy(gpu_storage_vec->a[i].str));
	}



}


void gasal_host_mem_free(gasal_gpu_storage_t *gpu_storage) {

	cudaError_t err;
	if (gpu_storage->host_unpacked1 != NULL) CHECKCUDAERROR(cudaFreeHost(gpu_storage->host_unpacked1));


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

//void gasal_host_malloc(void **mem_ptr, uint32_t n_bytes) {
//
//	cudaError_t err;
//	CHECKCUDAERROR(cudaMallocHost(mem_ptr, n_bytes));
//}
//
//void gasal_host_free(void *mem_free_ptr) {
//
//	cudaError_t err;
//	CHECKCUDAERROR(cudaFreeHost(mem_free_ptr));
//}



void gasal_host_malloc_uint32(uint32_t **mem_ptr, uint32_t n_bytes) {

	cudaError_t err;
	CHECKCUDAERROR(cudaMallocHost(mem_ptr, n_bytes));
}

void gasal_host_malloc_int32(int32_t **mem_ptr, uint32_t n_bytes) {

	cudaError_t err;
	CHECKCUDAERROR(cudaMallocHost(mem_ptr, n_bytes));
}

void gasal_host_malloc_uint8(uint8_t **mem_ptr, uint32_t n_bytes) {

	cudaError_t err;
	CHECKCUDAERROR(cudaMallocHost(mem_ptr, n_bytes));
}


void gasal_host_free_uint32(uint32_t *mem_free_ptr) {

	cudaError_t err;
	CHECKCUDAERROR(cudaFreeHost(mem_free_ptr));
}

void gasal_host_free_int32(int32_t *mem_free_ptr) {

	cudaError_t err;
	CHECKCUDAERROR(cudaFreeHost(mem_free_ptr));
}

void gasal_host_free_uint8(uint8_t *mem_free_ptr) {

	cudaError_t err;
	CHECKCUDAERROR(cudaFreeHost(mem_free_ptr));
}




