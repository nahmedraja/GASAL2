#include "gasal.h"
#include "args_parser.h"
#include "interfaces.h"
#include "host_batch.h"




// Functions for host batches handling. 

host_batch_t *gasal_host_batch_new(uint32_t batch_bytes, uint32_t offset)
{
	cudaError_t err;
	host_batch_t *res = (host_batch_t *)calloc(1, sizeof(host_batch_t));
	CHECKCUDAERROR(cudaHostAlloc(&(res->data), batch_bytes*sizeof(uint8_t), cudaHostAllocDefault));
	res->offset = offset;
	res->next = NULL;
	return res;
}

void gasal_host_batch_destroy(host_batch_t *res)
{
	cudaError_t err;
	if (res==NULL)
	{
		fprintf(stderr, "[GASAL ERROR] Trying to free a NULL pointer\n");
		exit(1);
	}
	// recursive function to destroy all the linked list
	if (res->next != NULL)
		gasal_host_batch_destroy(res->next);
	if (res->data != NULL) 
	{
		CHECKCUDAERROR(cudaFreeHost(res->data));
	}
	
	free(res);
}

host_batch_t *gasal_host_batch_getlast(host_batch_t *arg)
{
	return (arg->next == NULL ? arg : gasal_host_batch_getlast(arg->next) );
	
}


uint32_t gasal_host_batch_fill(gasal_gpu_storage_t *gpu_storage, uint32_t idx, const char* data, uint32_t size, data_source SRC)
{	
	// since query and target are very symmetric here, we use pointers to route the data where it has to, 
	// while keeping the actual memory management 'source-agnostic'.

	host_batch_t *cur_page = NULL;
	uint32_t *p_batch_bytes = NULL;

	switch(SRC) {
		case QUERY:
			cur_page = (gpu_storage->extensible_host_unpacked_query_batch);
			p_batch_bytes = &(gpu_storage->host_max_query_batch_bytes);
		break;
		case TARGET:
			cur_page = (gpu_storage->extensible_host_unpacked_target_batch);
			p_batch_bytes = &(gpu_storage->host_max_target_batch_bytes);
		break;
		default:
		break;
	}
	
	int nbr_N = 0;
	while((size+nbr_N)%8)
		nbr_N++;

	int is_done = 0;

	while (!is_done)
	{
		if (*p_batch_bytes < idx + size + nbr_N)
		{
			fprintf(stderr,"[GASAL WARNING:] Trying to write %d bytes at position %d on host memory (%s) while only  %d bytes are available. Therefore, allocating %d bytes more on CPU. Repeating this many times can provoke a degradation of performance.\n",
					size + nbr_N,
					idx,
					(SRC == QUERY ? "query":"target"),
					*p_batch_bytes,
					*p_batch_bytes * 2);
			
	
			*p_batch_bytes += *p_batch_bytes;

			// corner case: if we allocated less than a single sequence length to begin with... it shouldn't be allowed actually, but at least it's caught here.
			while (*p_batch_bytes < size)
				*p_batch_bytes += *p_batch_bytes;
			cur_page = gasal_host_batch_getlast(cur_page);
			host_batch_t *res = gasal_host_batch_new(*p_batch_bytes, idx);

			cur_page->next = res;
			
			cur_page = cur_page->next;

		} else if ((cur_page->next != NULL) && (cur_page->next->offset < idx + size + nbr_N)) {
			cur_page = cur_page->next;
			// if it's the first time you have to jump to the next page, reset idx - 
			// between recyclings, the offset can change for the TARGET in particular ; and if idx > offset, then you will try to write at a negative offset - that is, hit an unknown memory block
			if (cur_page->offset > idx)
				cur_page->offset = idx;
		} else {

			memcpy(&(cur_page->data[idx - cur_page->offset]), data, size);
	
			idx = idx + size;
	
			while(idx%8)
			{
				cur_page->data[idx - cur_page->offset] = N_CODE;
				idx++;
			}
			is_done = 1;
		}
		//gasal_host_batch_printall(gasal_host_batch_getlast(cur_page));

	}

	return idx;
}


uint32_t gasal_host_batch_addbase(gasal_gpu_storage_t *gpu_storage, uint32_t idx, const char base, data_source SRC )
{	 
    return gasal_host_batch_add(gpu_storage, idx, &base, 1, SRC );
}


uint32_t gasal_host_batch_add(gasal_gpu_storage_t *gpu_storage, uint32_t idx, const char *data, uint32_t size, data_source SRC )
{	

	// since query and target are very symmetric here, we use pointers to route the data where it has to, 
	// while keeping the actual memory management 'source-agnostic'.
	host_batch_t *cur_page = NULL;
	uint32_t *p_batch_bytes = NULL;
	

	switch(SRC) {
		case QUERY:
			cur_page = (gpu_storage->extensible_host_unpacked_query_batch);
			p_batch_bytes = &(gpu_storage->host_max_query_batch_bytes);
		break;
		case TARGET:
			cur_page = (gpu_storage->extensible_host_unpacked_target_batch);
			p_batch_bytes = &(gpu_storage->host_max_target_batch_bytes);
		break;
		default:
		break;
	}

	int is_done = 0;

	while (!is_done)
	{
		if (*p_batch_bytes >= idx + size && (cur_page->next == NULL || (cur_page->next->offset >= idx + size)) )
		{

			memcpy(&(cur_page->data[idx - cur_page->offset]), data, size);
			idx = idx + size;
			is_done = 1;

		} else if ((*p_batch_bytes >= idx + size) && (cur_page->next != NULL) && (cur_page->next->offset < idx + size)) {
		
			cur_page = cur_page->next;

		} else {
			fprintf(stderr,"[GASAL WARNING:] Trying to write %d bytes at position %d on host memory (%s) while only  %d bytes are available. Therefore, allocating %d bytes more on CPU. Repeating this many times can provoke a degradation of performance.\n",
					size,
					idx,
					(SRC == QUERY ? "query":"target"),
					*p_batch_bytes,
					*p_batch_bytes * 2);
			
	
			*p_batch_bytes += *p_batch_bytes;

			// corner case: if we allocated less than a single sequence length to begin with... it shouldn't be allowed actually, but at least it's caught here.
			while (*p_batch_bytes < size)
				*p_batch_bytes += *p_batch_bytes;

			host_batch_t *res = gasal_host_batch_new(*p_batch_bytes, idx);
	
			cur_page->next = res;
			
			cur_page = cur_page->next;
		}
	}
	//gasal_host_batch_printall(gasal_host_batch_getlast(cur_page));
	return idx;
}



// this printer displays the whole sequence. It is heavy and shouldn't be called when you have more than a couple sequences.
void gasal_host_batch_print(host_batch_t *res) 
{
	if (res->next != NULL)
		fprintf(stderr, "[GASAL PRINT] Page with offset %d, next page has offset %d\n",res->offset, (res->next->offset));
	else
		fprintf(stderr, "[GASAL PRINT] Page with offset %d, next page has offset NULL (last page)\n",res->offset);
}

// this printer allows to see the linked list easily.
void gasal_host_batch_printall(host_batch_t *res)
{
	int len = strlen((char*) res->data);
	fprintf(stderr, "[GASAL PRINT] Page data: offset=%d, next_offset=%d, data size=%d, data=%c%c%c%c...%c%c%c%c\n", res->offset, (res->next == NULL? -1 : (int)res->next->offset), (unsigned int)len, res->data[0], res->data[1], res->data[2], res->data[3], res->data[len-4], res->data[len-3], res->data[len-2], res->data[len-1]);
	if (res->next != NULL)
	{
		fprintf(stderr, "+--->");
		gasal_host_batch_printall(res->next);
	}
}
