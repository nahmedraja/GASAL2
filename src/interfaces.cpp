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
