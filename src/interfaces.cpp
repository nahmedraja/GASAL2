#include "gasal.h"
#include "interfaces.h"


void gasal_op_fill(gasal_gpu_storage_t *gpu_storage_t, uint8_t *data, uint32_t nbr_seqs_in_stream, data_source SRC)
{
	uint8_t *host_op = NULL;
	switch(SRC)
	{
		case QUERY:
			host_op = (gpu_storage_t->host_query_op);
		break;
		case TARGET:
			host_op = (gpu_storage_t->host_target_op);
		break;
		default:
		break;
	}
	memcpy(host_op, data, nbr_seqs_in_stream);
}


void gasal_set_device(int gpu_select)
{
	/* 
	Select GPU
	*/
	int num_devices, device;
	cudaGetDeviceCount(&num_devices);
	fprintf(stderr, "Found %d GPUs\n", num_devices);
	if (gpu_select  > num_devices-1)
	{
		fprintf(stderr, "Error: can't select device %d when only %d devices are selected (range from 0 to %d)\n", gpu_select, num_devices, num_devices-1);
		exit(EXIT_FAILURE);
	}
	if (num_devices > 0) {
		cudaDeviceProp properties;
		for (device = 0; device < num_devices; device++) {
				cudaGetDeviceProperties(&properties, device);
				fprintf(stderr, "\tGPU %d: %s\n", device, properties.name);
		}
		cudaGetDeviceProperties(&properties, gpu_select);
		fprintf(stderr, "Selected device %d : %s\n", gpu_select, properties.name);
		cudaSetDevice(gpu_select);
	}
}
