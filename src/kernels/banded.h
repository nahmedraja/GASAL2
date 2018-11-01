#ifndef KERNEL_BANDED
#define KERNEL_BANDED

#define BAND_SIZE (24)
#define __MOD(a) (a & (BAND_SIZE-1))


__global__ void gasal_banded_kernel(uint32_t *packed_query_batch, uint32_t *packed_target_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, gasal_res_t *device_res, int n_tasks, int32_t k_band_width) 
{
	int32_t i, j, k, m, l;
	int32_t e;
	int32_t maxHH = 0;//initialize the maximum score to zero
	int32_t prev_maxHH = 0;
	int32_t subScore;
	int32_t ridx, gidx;
	short2 HD;
	short2 initHD = make_short2(0, 0);
	int32_t maxXY_x = 0;
	int32_t maxXY_y = 0;
	const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;//thread ID
	if (tid >= n_tasks) return;
	uint32_t packed_target_batch_idx = target_batch_offsets[tid] >> 3; //starting index of the target_batch sequence
	uint32_t packed_query_batch_idx = query_batch_offsets[tid] >> 3;//starting index of the query_batch sequence
	uint32_t read_len = query_batch_lens[tid];
	uint32_t ref_len = target_batch_lens[tid];
	uint32_t query_batch_regs = (read_len >> 3) + (read_len&7 ? 1 : 0);//number of 32-bit words holding query_batch sequence
	uint32_t target_batch_regs = (ref_len >> 3) + (ref_len&7 ? 1 : 0);//number of 32-bit words holding target_batch sequence
	//-----arrays for saving intermediate values------
	short2 global[MAX_SEQ_LEN];
	int32_t h[9];
	int32_t f[9];
	int32_t p[9];
	int32_t k_other_band_width = (target_batch_regs*8 - (query_batch_regs*8 - k_band_width));
	//--------------------------------------------

	// table of cells (don't use it with sequences larger than ~50 bases)
	/*
	#ifdef DEBUG
	if (tid==0) {
		for (j = 1; j <= query_batch_regs*8; j+=1) {
			printf("%03d\t", j);
			for (i = 1; i <= target_batch_regs*8; i++) {
				int x = i;
				int y = j;
				if (y > k_band_width + x || x > y + (target_batch_regs*8 - (query_batch_regs*8 - k_band_width)))
				{
					printf("_");
				} else {
					printf("#");
				}
				if (i%8 == 0)
					printf(" ");
			}
			printf("\n");
			if (j%8 == 0)
				printf("\n");
			
		}
	}
	#endif
	*/
	

	//------------------------
	for (i = 0; i < MAX_SEQ_LEN; i++) {
		global[i] = initHD;
	}


	for (i = 0; i < target_batch_regs; i++) { //target_batch sequence in rows
		for (m = 0; m < 9; m++) {
			h[m] = 0;
			f[m] = 0;
			p[m] = 0;
		}
		register uint32_t gpac =packed_target_batch[packed_target_batch_idx + i];//load 8 packed bases from target_batch sequence
		gidx = i << 3;
		ridx = 0;

		for (j = 0; j < query_batch_regs; j++) { //query_batch sequence in columns
			register uint32_t rpac =packed_query_batch[packed_query_batch_idx + j];//load 8 bases from query_batch sequence

			//--------------compute a tile of 8x8 cells-------------------
			for (k = 28; k >= 0; k -= 4) {
				register uint32_t rbase = (rpac >> k) & 0x0F;//get a base from query_batch sequence
				//-----load intermediate values--------------
				HD = global[ridx];
				h[0] = HD.x;
				e = HD.y;
				//-------------------------------------------
				//int32_t prev_hm_diff = h[0] - _cudaGapOE;

				#pragma unroll 8
				for (l = 28, m = 1; m < 9; l -= 4, m++) {
					// let x,y be the coordinates of the cell in the DP matrix.
					int32_t x_minus_y = ((i) << 3) + (((28-k)>>2) - (((j) << 3) + ((28-l)>>2)));
					/*
					// display ALL CELLS
					if (tid==0)
						printf("(%d, %d) - ", x, y);
					*/

					if (-x_minus_y > k_band_width || x_minus_y > k_other_band_width)
					{

						h[m] = 0;

					} else {
					register uint32_t gbase = (gpac >> l) & 15;//get a base from target_batch sequence
					DEV_GET_SUB_SCORE_LOCAL(subScore, rbase, gbase);//check equality of rbase and gbase
					//int32_t curr_hm_diff = h[m] - _cudaGapOE;
					f[m] = max(h[m]- _cudaGapOE, f[m] - _cudaGapExtend);//whether to introduce or extend a gap in query_batch sequence
					h[m] = p[m] + subScore;//score if rbase is aligned to gbase
					h[m] = max(h[m], f[m]);
					h[m] = max(h[m], 0);
					e = max(h[m - 1] - _cudaGapOE, e - _cudaGapExtend);//whether to introduce or extend a gap in target_batch sequence
					//prev_hm_diff=curr_hm_diff;
					h[m] = max(h[m], e);
					
					
					//FIND_MAX(h[m], gidx + (m-1))//the current maximum score and corresponding end position on target_batch sequence
					maxXY_y = (maxHH < h[m]) ? (gidx + (m-1)) : maxXY_y;
					maxHH = (maxHH < h[m]) ? h[m] : maxHH;

					p[m] = h[m-1];
					}
				}
				//----------save intermediate values------------
				HD.x = h[m-1];
				HD.y = e;
				global[ridx] = HD;
				//---------------------------------------------
				maxXY_x = (prev_maxHH < maxHH) ? ridx : maxXY_x;//end position on query_batch sequence corresponding to current maximum score
				prev_maxHH = max(maxHH, prev_maxHH);
				ridx++;
			}
			//-------------------------------------------------------
			// 8*8 patch done

		}

	}

	device_res->aln_score[tid] = maxHH;//copy the max score to the output array in the GPU mem
	device_res->query_batch_end[tid] = maxXY_x;//copy the end position on query_batch sequence to the output array in the GPU mem
	device_res->target_batch_end[tid] = maxXY_y;//copy the end position on target_batch sequence to the output array in the GPU mem

	return;

}

__global__ void gasal_banded_with_start_kernel(uint32_t *packed_query_batch, uint32_t *packed_target_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, int32_t *score, int32_t *query_batch_end, int32_t *target_batch_end, int32_t *query_batch_start, int32_t *target_batch_start,int n_tasks, int32_t k_band_width) 
{
	int32_t i, j, k, m, l;
	int32_t e;
	int32_t maxHH = 0;//initialize the maximum score to zero
	int32_t prev_maxHH = 0;
	int32_t subScore;
	int32_t ridx, gidx;
	short2 HD;
	short2 initHD = make_short2(0, 0);
	int32_t maxXY_x = 0;
	int32_t maxXY_y = 0;
	const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;//thread ID
	if (tid >= n_tasks) return;
	uint32_t packed_target_batch_idx = target_batch_offsets[tid] >> 3; //starting index of the target_batch sequence
	uint32_t packed_query_batch_idx = query_batch_offsets[tid] >> 3;//starting index of the query_batch sequence
	uint32_t read_len = query_batch_lens[tid];
	uint32_t ref_len = target_batch_lens[tid];
	uint32_t query_batch_regs = (read_len >> 3) + (read_len&7 ? 1 : 0);//number of 32-bit words holding query_batch sequence
	uint32_t target_batch_regs = (ref_len >> 3) + (ref_len&7 ? 1 : 0);//number of 32-bit words holding target_batch sequence
	//-----arrays for saving intermediate values------
	short2 global[MAX_SEQ_LEN];
	int32_t h[9];
	int32_t f[9];
	int32_t p[9];
	//--------------------------------------------

	// table of cells (don't use it with sequences larger than ~50 bases)
	/*
	#ifdef DEBUG
	if (tid==0) {
		for (j = 1; j <= query_batch_regs*8; j+=1) {
			printf("%03d\t", j);
			for (i = 1; i <= target_batch_regs*8; i++) {
				int x = i;
				int y = j;
				if (y > k_band_width + x || x > y + (target_batch_regs*8 - (query_batch_regs*8 - k_band_width)))
				{
					printf("_");
				} else {
					printf("#");
				}
				if (i%8 == 0)
					printf(" ");
			}
			printf("\n");
			if (j%8 == 0)
				printf("\n");
			
		}
	}
	#endif
	*/
	

	//------------------------
	for (i = 0; i < MAX_SEQ_LEN; i++) {
		global[i] = initHD;
	}


	for (i = 0; i < target_batch_regs; i++) { //target_batch sequence in rows
		for (m = 0; m < 9; m++) {
			h[m] = 0;
			f[m] = 0;
			p[m] = 0;
		}
		register uint32_t gpac =packed_target_batch[packed_target_batch_idx + i];//load 8 packed bases from target_batch sequence
		gidx = i << 3;
		ridx = 0;

		for (j = 0; j < query_batch_regs; j++) { //query_batch sequence in columns
			register uint32_t rpac =packed_query_batch[packed_query_batch_idx + j];//load 8 bases from query_batch sequence

			//--------------compute a tile of 8x8 cells-------------------
			for (k = 28; k >= 0; k -= 4) {
				uint32_t rbase = (rpac >> k) & 0x0F;//get a base from query_batch sequence
				//-----load intermediate values--------------
				HD = global[ridx];
				h[0] = HD.x;
				e = HD.y;
				//-------------------------------------------
				//int32_t prev_hm_diff = h[0] - _cudaGapOE;

				#pragma unroll 8
				for (l = 28, m = 1; m < 9; l -= 4, m++) {
					// let x,y be the coordinates of the cell in the DP matrix.
					int32_t x = ((i) << 3) + ((28-k)>>2);
					int32_t y = ((j) << 3) + ((28-l)>>2);
					/*
					// display ALL CELLS
					if (tid==0)
						printf("(%d, %d) - ", x, y);
					*/

					if (y > k_band_width + x || x > y + (target_batch_regs*8 - (query_batch_regs*8 - k_band_width)))
					{
						h[m] = 0;

					} else {
					uint32_t gbase = (gpac >> l) & 15;//get a base from target_batch sequence
					DEV_GET_SUB_SCORE_LOCAL(subScore, rbase, gbase);//check equality of rbase and gbase
					//int32_t curr_hm_diff = h[m] - _cudaGapOE;
					f[m] = max(h[m]- _cudaGapOE, f[m] - _cudaGapExtend);//whether to introduce or extend a gap in query_batch sequence
					h[m] = p[m] + subScore;//score if rbase is aligned to gbase
					h[m] = max(h[m], f[m]);
					h[m] = max(h[m], 0);
					e = max(h[m - 1] - _cudaGapOE, e - _cudaGapExtend);//whether to introduce or extend a gap in target_batch sequence
					//prev_hm_diff=curr_hm_diff;
					h[m] = max(h[m], e);
						
					FIND_MAX(h[m], gidx + (m-1))//the current maximum score and corresponding end position on target_batch sequence
					p[m] = h[m-1];
					}
				}
				//----------save intermediate values------------
				HD.x = h[m-1];
				HD.y = e;
				global[ridx] = HD;
				//---------------------------------------------
				maxXY_x = (prev_maxHH < maxHH) ? ridx : maxXY_x;//end position on query_batch sequence corresponding to current maximum score
				prev_maxHH = max(maxHH, prev_maxHH);
				ridx++;
			}
			//-------------------------------------------------------
			// 8*8 patch done

		}

	}

	score[tid] = maxHH;//copy the max score to the output array in the GPU mem
	query_batch_end[tid] = maxXY_x;//copy the end position on query_batch sequence to the output array in the GPU mem
	target_batch_end[tid] = maxXY_y;//copy the end position on target_batch sequence to the output array in the GPU mem

	/*------------------Now to find the start position-----------------------*/

	int32_t rend_pos = maxXY_x;//end position on query_batch sequence
	int32_t gend_pos = maxXY_y;//end position on target_batch sequence
	int32_t fwd_score = maxHH;// the computed score


	int32_t rend_reg = ((rend_pos >> 3) + 1) < query_batch_regs ? ((rend_pos >> 3) + 1) : query_batch_regs;//the index of 32-bit word containing the end position on query_batch sequence
	int32_t gend_reg = ((gend_pos >> 3) + 1) < target_batch_regs ? ((gend_pos >> 3) + 1) : target_batch_regs;//the index of 32-bit word containing to end position on target_batch sequence


	packed_query_batch_idx += (rend_reg - 1);
	packed_target_batch_idx += (gend_reg - 1);


	maxHH = 0;
	prev_maxHH = 0;
	maxXY_x = 0;
	maxXY_y = 0;

	for (i = 0; i < MAX_SEQ_LEN; i++) {
		global[i] = initHD;
	}
	//------starting from the gend_reg and rend_reg, align the sequences in the reverse direction and exit if the max score >= fwd_score------
	gidx = ((gend_reg << 3) + 8) - 1;
	for (i = 0; i < gend_reg && maxHH < fwd_score; i++) {
		for (m = 0; m < 9; m++) {
			h[m] = 0;
			f[m] = 0;
			p[m] = 0;
		}
		register uint32_t gpac =packed_target_batch[packed_target_batch_idx - i];//load 8 packed bases from target_batch sequence
		gidx = gidx - 8;
		ridx = (rend_reg << 3) - 1;
		int32_t global_idx = 0;
		for (j = 0; j < rend_reg && maxHH < fwd_score; j+=1) {
			register uint32_t rpac =packed_query_batch[packed_query_batch_idx - j];//load 8 packed bases from query_batch sequence
			//--------------compute a tile of 8x8 cells-------------------
			for (k = 0; k <= 28 && maxHH < fwd_score; k += 4) {
				uint32_t rbase = (rpac >> k) & 15;//get a base from query_batch sequence
				//----------load intermediate values--------------
				HD = global[global_idx];
				h[0] = HD.x;
				e = HD.y;
				//-----------------------------------------------
	#pragma unroll 8
				for (l = 0, m = 1; l <= 28; l += 4, m++) {
					// let x,y be the coordinates of the cell in the DP matrix.
					int32_t x = ((i) << 3) + ((28-k)>>2);
					int32_t y = ((j) << 3) + ((28-l)>>2);
					/*
					// display ALL CELLS
					if (tid==0)
						printf("(%d, %d) - ", x, y);
					*/

					if (y > k_band_width + x || x > y + (target_batch_regs*8 - (query_batch_regs*8 - k_band_width)))
					{
						h[m] = 0;

					} else {
						uint32_t gbase = (gpac >> l) & 15;//get a base from target_batch sequence
						DEV_GET_SUB_SCORE_LOCAL(subScore, rbase, gbase);//check equality of rbase and gbase
						f[m] = max(h[m]- _cudaGapOE, f[m] - _cudaGapExtend);//whether to introduce or extend a gap in query_batch sequence

						h[m] = p[m] + subScore;//score if gbase is aligned to rbase
						h[m] = max(h[m], f[m]);
						h[m] = max(h[m], 0);
						e = max(h[m-1] - _cudaGapOE, e - _cudaGapExtend);//whether to introduce or extend a gap in target_batch sequence
						h[m] = max(h[m], e);

						FIND_MAX(h[m], gidx - (m -1));//the current maximum score and corresponding start position on target_batch sequence
						p[m] = h[m-1];
					}
				}
				//------------save intermediate values----------------
				HD.x = h[m-1];
				HD.y = e;
				global[global_idx] = HD;
				//----------------------------------------------------
				maxXY_x = (prev_maxHH < maxHH) ? ridx : maxXY_x;//start position on query_batch sequence corresponding to current maximum score
				prev_maxHH = max(maxHH, prev_maxHH);
				ridx--;
				global_idx++;
			}
			//-------------------------------------------------------



		}

	}
	//------------------------------------------------------------------------------------------------------------------------------------

	query_batch_start[tid] = maxXY_x;//copy the start position on query_batch sequence to the output array in the GPU mem
	target_batch_start[tid] = maxXY_y;//copy the start position on target_batch sequence to the output array in the GPU mem

	return;

}


__global__ void gasal_banded_tiled_kernel(uint32_t *packed_query_batch, uint32_t *packed_target_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, gasal_res_t *device_res, int n_tasks, const int32_t k_band_width) 
{
	int32_t i, j, k, m, l;
	int32_t e;
	int32_t maxHH = 0;//initialize the maximum score to zero
	int32_t prev_maxHH = 0;
	int32_t subScore;
	int32_t ridx, gidx;
	short2 HD;
	short2 initHD = make_short2(0, 0);
	int32_t maxXY_x = 0;
	int32_t maxXY_y = 0;
	const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;//thread ID
	if (tid >= n_tasks) return;
	uint32_t packed_target_batch_idx = target_batch_offsets[tid] >> 3; //starting index of the target_batch sequence
	uint32_t packed_query_batch_idx = query_batch_offsets[tid] >> 3;//starting index of the query_batch sequence
	uint32_t read_len = query_batch_lens[tid];
	uint32_t ref_len = target_batch_lens[tid];
	uint32_t query_batch_regs = (read_len >> 3) + (read_len&7 ? 1 : 0);//number of 32-bit words holding query_batch sequence
	uint32_t target_batch_regs = (ref_len >> 3) + (ref_len&7 ? 1 : 0);//number of 32-bit words holding target_batch sequence
	//-----arrays for saving intermediate values------
	short2 global[MAX_SEQ_LEN];
	int32_t h[9];
	int32_t f[9];
	int32_t p[9];
 	const int32_t k_other_band_width = (target_batch_regs - (query_batch_regs - k_band_width));

	//--------------------------------------------

	// table of cells (don't use it with sequences larger than ~50 bases)
	/*
		#ifdef DEBUG
		if (tid==0) {
			for (j = 1; j <= query_batch_regs*8; j+=1) {
				printf("%03d\t", j);
				for (i = 1; i <= target_batch_regs*8; i++) {
					int x = i;
					int y = j;
					if (y > k_band_width + x || x > y + (target_batch_regs*8 - (query_batch_regs*8 - k_band_width)))
					{
						printf("_");
					} else {
						printf("#");
					}
					if (i%8 == 0)
						printf(" ");
				}
				printf("\n");
				if (j%8 == 0)
					printf("\n");
				
			}
		}
		#endif
	*/
	

	//------------------------
	for (i = 0; i < MAX_SEQ_LEN; i++) {
		global[i] = initHD;
	}


	for (i = 0; i < target_batch_regs; i++) { //target_batch sequence in rows
		for (m = 0; m < 9; m++) {
			h[m] = 0;
			f[m] = 0;
			p[m] = 0;
		}
		
		uint32_t gpac =packed_target_batch[packed_target_batch_idx + i];//load 8 packed bases from target_batch sequence
		gidx = i << 3;

		ridx = MAX(0, i - k_other_band_width+1) << 3;
		int32_t last_tile =  MIN( k_band_width + i, (int32_t)query_batch_regs);
		for (j = ridx >> 3  ; j < last_tile; j++) { //query_batch sequence in columns --- the beginning and end are defined with the tile-based band, to avoid unneccessary calculations.

			uint32_t rpac =packed_query_batch[packed_query_batch_idx + j];//load 8 bases from query_batch sequence

				//--------------compute a tile of 8x8 cells-------------------
				for (k = 28; k >= 0; k -= 4) {
					uint32_t rbase = (rpac >> k) & 0x0F;//get a base from query_batch sequence
					//-----load intermediate values--------------
					HD = global[ridx];
					h[0] = HD.x;
					e = HD.y;
					//-------------------------------------------
					//int32_t prev_hm_diff = h[0] - _cudaGapOE;

					#pragma unroll 8
					for (l = 28, m = 1; m < 9; l -= 4, m++) {
						uint32_t gbase = (gpac >> l) & 15;//get a base from target_batch sequence
						DEV_GET_SUB_SCORE_LOCAL(subScore, rbase, gbase);//check equality of rbase and gbase
						//int32_t curr_hm_diff = h[m] - _cudaGapOE;
						f[m] = max(h[m]- _cudaGapOE, f[m] - _cudaGapExtend);//whether to introduce or extend a gap in query_batch sequence
						h[m] = p[m] + subScore;//score if rbase is aligned to gbase
						h[m] = max(h[m], f[m]);
						h[m] = max(h[m], 0);
						e = max(h[m - 1] - _cudaGapOE, e - _cudaGapExtend);//whether to introduce or extend a gap in target_batch sequence
						//prev_hm_diff=curr_hm_diff;
						h[m] = max(h[m], e);
						
						FIND_MAX(h[m], gidx + (m-1))//the current maximum score and corresponding end position on target_batch sequence

						p[m] = h[m-1];
					}
					//----------save intermediate values------------
					HD.x = h[m-1];
					HD.y = e;
					global[ridx] = HD;
					//---------------------------------------------
					maxXY_x = (prev_maxHH < maxHH) ? ridx : maxXY_x;//end position on query_batch sequence corresponding to current maximum score
					prev_maxHH = max(maxHH, prev_maxHH);
					ridx++;
				
			}
			//-------------------------------------------------------
			// 8*8 patch done

		}

	}

	device_res->aln_score[tid] = maxHH;//copy the max score to the output array in the GPU mem
	device_res->query_batch_end[tid] = maxXY_x;//copy the end position on query_batch sequence to the output array in the GPU mem
	device_res->target_batch_end[tid] = maxXY_y;//copy the end position on target_batch sequence to the output array in the GPU mem

	return;

}
/*
__global__ void gasal_banded_fixed_kernel(uint32_t *packed_query_batch, uint32_t *packed_target_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, int32_t *score, int32_t *query_batch_end, int32_t *target_batch_end, int n_tasks) {
	int32_t i, j, k, m, l;
	int32_t e;
	int32_t maxHH = 0;//initialize the maximum score to zero
	int32_t prev_maxHH = 0;
	int32_t subScore;
	int32_t ridx, gidx;
	short2 HD;
	short2 initHD = make_short2(0, 0);
	int32_t maxXY_x = 0;
	int32_t maxXY_y = 0;
	const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;//thread ID
	if (tid >= n_tasks) return;
	uint32_t packed_target_batch_idx = target_batch_offsets[tid] >> 3; //starting index of the target_batch sequence
	uint32_t packed_query_batch_idx = query_batch_offsets[tid] >> 3;//starting index of the query_batch sequence
	uint32_t read_len = query_batch_lens[tid];
	//uint32_t ref_len = target_batch_lens[tid];  // - unused in case of square matrix computation.
	uint32_t query_batch_regs = (read_len >> 3) + (read_len&7 ? 1 : 0);//number of 32-bit words holding query_batch sequence 
	//uint32_t target_batch_regs = (ref_len >> 3) + (ref_len&7 ? 1 : 0);//number of 32-bit words holding target_batch sequence - unused in case of square matrix computation.
	//-----arrays for saving intermediate values------
	short2 global[BAND_SIZE];
	int32_t h[9];
	int32_t f[9];
	int32_t p[9];


	//------------------------
	for (i = 0; i < BAND_SIZE; i++) {
		global[i] = initHD;
	}


	for (i = 0; i < query_batch_regs; i++) { //QUERY instead of TARGET because we're only calculating as if it was a SQUARE MATRIX. --- target_batch sequence in rows
		for (m = 0; m < 9; m++) {
			h[m] = 0;
			f[m] = 0;
			p[m] = 0;
		}

		register uint32_t gpac =packed_target_batch[packed_target_batch_idx + i];//load 8 packed bases from target_batch sequence
		gidx = i << 3;
		ridx = __MOD(i<<3);
		
		#pragma unroll 8
		for (int b = 0; b < 8; b++)
		{
			global[__MOD((i-1)<<3)] = initHD;
		}
		
		
		for (j = MIN(i, BAND_SIZE) ; j < BAND_SIZE ; j++ ) { //query_batch sequence in columns
			register uint32_t rpac =packed_query_batch[packed_query_batch_idx + j];//load 8 bases from query_batch sequence

			//--------------compute a tile of 8x8 cells-------------------
			for (k = 28; k >= 0; k -= 4) {
				register uint32_t rbase = (rpac >> k) & 0x0F;//get a base from query_batch sequence
				//-----load intermediate values--------------
				HD = global[__MOD(ridx)];
				h[0] = HD.x;
				e = HD.y;
				//-------------------------------------------
				//int32_t prev_hm_diff = h[0] - _cudaGapOE;

				#pragma unroll 8
				for (l = 28, m = 1; m < 9; l -= 4, m++) {
					
					register uint32_t gbase = (gpac >> l) & 15;//get a base from target_batch sequence
					DEV_GET_SUB_SCORE_LOCAL(subScore, rbase, gbase);//check equality of rbase and gbase
					//int32_t curr_hm_diff = h[m] - _cudaGapOE;
					f[m] = max(h[m]- _cudaGapOE, f[m] - _cudaGapExtend);//whether to introduce or extend a gap in query_batch sequence
					h[m] = p[m] + subScore;//score if rbase is aligned to gbase
					h[m] = max(h[m], f[m]);
					h[m] = max(h[m], 0);
					e = max(h[m - 1] - _cudaGapOE, e - _cudaGapExtend);//whether to introduce or extend a gap in target_batch sequence
					//prev_hm_diff=curr_hm_diff;
					h[m] = max(h[m], e);
				
					FIND_MAX(h[m], gidx + (m-1))//the current maximum score and corresponding end position on target_batch sequence

					p[m] = h[m-1];
					
				}
				//----------save intermediate values------------
				HD.x = h[m-1];
				HD.y = e;
				global[ridx] = HD;
				//---------------------------------------------
				maxXY_x = (prev_maxHH < maxHH) ? ((j<<3)+__MOD(ridx)) : maxXY_x;//end position on query_batch sequence corresponding to current maximum score
				prev_maxHH = max(maxHH, prev_maxHH);
				ridx++;
			}
			//-------------------------------------------------------
			// 8*8 patch done
		}

	}

	score[tid] = maxHH;//copy the max score to the output array in the GPU mem
	query_batch_end[tid] = maxXY_x;//copy the end position on query_batch sequence to the output array in the GPU mem
	target_batch_end[tid] = maxXY_y;//copy the end position on target_batch sequence to the output array in the GPU mem

	return;

}
*/
#endif