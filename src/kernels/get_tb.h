#ifndef __GET_TB__
#define __GET_TB__

template <typename T>
__global__ void gasal_get_tb(uint8_t *cigar, uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *cigar_offset, uint4 *packed_tb_matrices, gasal_res_t *device_res, int n_tasks) {

	int i, j;
	int total_score __attribute__((unused));
	int curr_score __attribute__((unused));
	const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n_tasks) return;

	int offset = cigar_offset[tid];


	if (SAMETYPE(T, Int2Type<LOCAL>)) {
		i = device_res->target_batch_end[tid];
		j = device_res->query_batch_end[tid];
		total_score = device_res->aln_score[tid];
		curr_score = 0;
	} else if (SAMETYPE(T, Int2Type<GLOBAL>)) {
		i = target_batch_lens[tid];
		j = query_batch_lens[tid];
	}



	uint32_t prev_op_to_fill = 0;

	int read_len_8 = query_batch_lens[tid]%8 ? query_batch_lens[tid] + (8 - (query_batch_lens[tid]%8)) : query_batch_lens[tid];

	int n_ops = 0;

	int prev_tile_no = -1;

	uint4 tile = make_uint4(0, 0, 0, 0);

	int op_select = 3;

	int op_shift = 0;


	int count = 0;

	uint32_t op_to_fill;

	while ( i >= 0 && j >= 0) {


		int cell = (((i >> 3) * read_len_8) << 3) + (j << 3) + (i&7);



		int tile_no = cell>>5;


		tile = tile_no != prev_tile_no ? packed_tb_matrices[(tile_no*n_tasks) + tid] : tile;

		prev_tile_no = tile_no;

		int cell_no_in_tile = cell - (tile_no<<5);


		int reg_no_in_tile = cell_no_in_tile >> 3;

		int cell_no_in_reg = cell_no_in_tile - (reg_no_in_tile << 3);

		uint32_t reg = reg_no_in_tile == 0 ? tile.x : (reg_no_in_tile == 1 ? tile.y : (reg_no_in_tile == 2 ? tile.z : tile.w));


		uint32_t cell_op = (reg >> (28 - (cell_no_in_reg << 2))) & 15;


		uint32_t op = (cell_op >> op_shift) & op_select;



		op_to_fill = op == 0 || op_select == 3 ? op : op_shift ;

		op_select = op == 0 || (op == 1 && op_select == 3) ? 3 : 1;

		op_shift = op == 0 || ( op == 1 && op_select == 3) ? 0 : ((op == 2 || op == 3) ?  op : op_shift);




		if(count < 63  &&  op_to_fill == prev_op_to_fill) {
			count++;
		} else {
			if (count > 0) {
				uint8_t reg_out = 0;
				reg_out |= prev_op_to_fill;
				reg_out |= (uint8_t)(count << 2);
				cigar[offset++] = reg_out;
				n_ops++;
			}
			count = 1;
		}

		if (SAMETYPE(T, Int2Type<LOCAL>)) {
			curr_score += ((op_to_fill == 2 || op_to_fill == 3) && prev_op_to_fill != op_to_fill) ? -_cudaGapOE : ((op_to_fill == 2 || op_to_fill == 3) ? - _cudaGapExtend : (op_to_fill == 1 ? -_cudaMismatchScore : _cudaMatchScore));
			if (curr_score == total_score) break;
		}

		prev_op_to_fill = op_to_fill;

		i = op_to_fill == 0 || op_to_fill == 1 || op_to_fill == 2 ? i - 1 : i;
		j = op_to_fill == 0 || op_to_fill == 1 || op_to_fill == 3 ? j - 1 : j;


	}

	uint8_t reg_out = 0;
	reg_out |= prev_op_to_fill;
	reg_out |= (uint8_t)(count << 2);
	cigar[offset++] = reg_out;
	n_ops++;

	if (SAMETYPE(T, Int2Type<GLOBAL>)) {
		while (i >= 0) {
			uint32_t reg_out = 0;
			uint8_t resd_count = (i+1) <= 63 ? (i+1) : 63;
			reg_out |= 2;
			reg_out |= (uint8_t)(resd_count << 2);
			cigar[offset++] = reg_out;
			n_ops++;
			i = i - 63;

		}
		while (j >= 0) {
			uint32_t reg_out = 0;
			uint8_t resd_count = (j+1) <= 63 ? (j+1) : 63;
			reg_out |= 3;
			reg_out |= (uint8_t)(resd_count << 2);
			cigar[offset++] = reg_out;
			n_ops++;
			j = j - 63;
		}
	}


	if (SAMETYPE(T, Int2Type<LOCAL>)) {
		device_res->target_batch_start[tid] = i;
		device_res->query_batch_start[tid] = j;
	}
	query_batch_lens[tid] = n_ops;


}
#endif
