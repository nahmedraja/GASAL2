#ifndef __HOST_BACTH_H__
#define __HOST_BACTH_H__

// host data structure methods
host_batch_t *gasal_host_batch_new(uint32_t host_max_query_batch_bytes, uint32_t offset); 								// constructor
void gasal_host_batch_destroy(host_batch_t *res); 																		// destructor
host_batch_t *gasal_host_batch_getlast(host_batch_t *arg); 																// get last item of chain
uint32_t gasal_host_batch_fill(gasal_gpu_storage_t *gpu_storage_t, uint32_t idx, const char* data, uint32_t size, data_source SRC ); 	// fill the data
void gasal_host_batch_print(host_batch_t *res); 																		// printer 
void gasal_host_batch_printall(host_batch_t *res);																		// printer for the whole linked list
void gasal_host_batch_recycle(gasal_gpu_storage_t *gpu_storage_t);


#endif
