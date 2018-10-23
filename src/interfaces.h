#ifndef __GASAL_INTERFACES_H__
#define __GASAL_INTERFACES_H__


// operation filler method (field in the gasal_gpu_storage_t field)
void gasal_op_fill(gasal_gpu_storage_t *gpu_storage_t, uint8_t *data, uint32_t nbr_seqs_in_stream, data_source SRC);


#endif
