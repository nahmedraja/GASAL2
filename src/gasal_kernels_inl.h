


__global__ void gasal_pack_kernel_4bit(uint32_t* batch1,
		uint32_t* batch2, uint32_t *batch1_4bit, uint32_t* batch2_4bit,
		int batch1_tasks_per_thread, int batch2_tasks_per_thread, uint32_t total_batch1_regs, uint32_t total_batch2_regs) {

	int32_t i;
	const int32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;//thread ID
	uint32_t n_threads = gridDim.x * blockDim.x;
	for (i = 0; i < batch1_tasks_per_thread &&  (((i*n_threads)<<1) + (tid<<1) < total_batch1_regs); ++i) {
		uint32_t *batch1_batch_addr = &(batch1[(i*n_threads)<<1]);
		uint32_t reg1 = batch1_batch_addr[(tid << 1)]; //load 4 bases of the first sequence from global memory
		uint32_t reg2 = batch1_batch_addr[(tid << 1) + 1]; //load  another 4 bases of the S1 from global memory
		uint32_t pack_reg_4bit = 0;
		pack_reg_4bit |= (reg1 & 15) << 28;        // ---
		pack_reg_4bit |= ((reg1 >> 8) & 15) << 24; //    |
		pack_reg_4bit |= ((reg1 >> 16) & 15) << 20;//    |
		pack_reg_4bit |= ((reg1 >> 24) & 15) << 16;//    |
		pack_reg_4bit |= (reg2 & 15) << 12;        //     > pack data
		pack_reg_4bit |= ((reg2 >> 8) & 15) << 8;  //    |
		pack_reg_4bit |= ((reg2 >> 16) & 15) << 4; //    |
		pack_reg_4bit |= ((reg2 >> 24) & 15);      //----
		uint32_t *batch1_4bit_batch_addr = &(batch1_4bit[i*n_threads]);
		batch1_4bit_batch_addr[tid] = pack_reg_4bit; // write 8 bases of S1 packed into a unsigned 32 bit integer to global memory
	}

	for (i = 0; i < batch2_tasks_per_thread &&  (((i*n_threads)<<1) + (tid<<1)) < total_batch2_regs; ++i) {
		uint32_t *batch2_batch_addr = &(batch2[(i * n_threads)<<1]);
		uint32_t reg1 = batch2_batch_addr[(tid << 1)]; //load 4 bases of the S2 from global memory
		uint32_t reg2 = batch2_batch_addr[(tid << 1) + 1]; //load  another 4 bases of the S2 from global memory
		uint32_t pack_reg_4bit = 0;
		pack_reg_4bit |= (reg1 & 15) << 28;        // ---
		pack_reg_4bit |= ((reg1 >> 8) & 15) << 24; //    |
		pack_reg_4bit |= ((reg1 >> 16) & 15) << 20;//    |
		pack_reg_4bit |= ((reg1 >> 24) & 15) << 16;//    |
		pack_reg_4bit |= (reg2 & 15) << 12;        //     > pack data
		pack_reg_4bit |= ((reg2 >> 8) & 15) << 8;  //    |
		pack_reg_4bit |= ((reg2 >> 16) & 15) << 4; //    |
		pack_reg_4bit |= ((reg2 >> 24) & 15);      //----
		uint32_t *batch2_4bit_batch_addr = &(batch2_4bit[i * n_threads]); //write 8 bases of S2 packed into a uint32_t to global memory
		batch2_4bit_batch_addr[tid] = pack_reg_4bit; // write 8 bases of S2 packed into a unsigned 32 bit integer to global memory
	}

}

__constant__ int32_t _cudaGapO; /*gap open penality*/
__constant__ int32_t _cudaGapOE; /*sum of gap open and extension penalites*/
__constant__ int32_t _cudaGapExtend; /*sum of gap extend*/
__constant__ int32_t _cudaMatchScore; /*score for a match*/
__constant__ int32_t _cudaMismatchScore; /*penalty for a mismatch*/

#define MINUS_INF SHRT_MIN

#ifdef N_SCORE
#define DEV_GET_SUB_SCORE(score, rbase, gbase) \
      	score = (rbase == gbase) ?_cudaMatchScore : -_cudaMismatchScore;\
        score = ((rbase == N) || (gbase == N)) ? N_SCORE : score;\

#else
#define DEV_GET_SUB_SCORE(score, rbase, gbase) \
      	score = (rbase == gbase) ?_cudaMatchScore : -_cudaMismatchScore;\

#endif


#define FIND_MAX(curr, gidx) \
								maxXY_y = (maxHH < curr) ? gidx : maxXY_y;\
								maxHH = (maxHH < curr) ? curr : maxHH;


__global__ void gasal_local_kernel(uint32_t *packed_batch1, uint32_t *packed_batch2,  uint32_t *batch1_len, uint32_t *batch2_len, uint32_t *batch1_offset, uint32_t *batch2_offset, int32_t *score, int32_t *batch1_end, int32_t *batch2_end, int n_tasks) {
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
		uint32_t packed_batch2_idx = batch2_offset[tid] >> 3; //starting index of the batch2 sequence
		uint32_t packed_batch1_idx = batch1_offset[tid] >> 3;//starting index of the batch1 sequence
		uint32_t read_len = batch1_len[tid];
		uint32_t ref_len = batch2_len[tid];
		uint32_t batch1_regs = (read_len >> 3) + (read_len&7 ? 1 : 0);//number of 32-bit words holding batch1 sequence
		uint32_t batch2_regs = (ref_len >> 3) + (ref_len&7 ? 1 : 0);//number of 32-bit words holding batch2 sequence
		//-----arrays for saving intermediate values------
		short2 global[MAX_SEQ_LEN];
		int32_t h[9];
		int32_t f[9];
		int32_t p[9];
		//--------------------------------------------
		for (i = 0; i < MAX_SEQ_LEN; i++) {
			global[i] = initHD;
		}


		for (i = 0; i < batch2_regs; i++) { //batch2 sequence in rows
			for (m = 0; m < 9; m++) {
					h[m] = 0;
					f[m] = 0;
					p[m] = 0;
			}
			register uint32_t gpac =packed_batch2[packed_batch2_idx + i];//load 8 packed bases from batch2 sequence
			gidx = i << 3;
			ridx = 0;
			for (j = 0; j < batch1_regs; j+=1) { //batch1 sequence in columns
				register uint32_t rpac =packed_batch1[packed_batch1_idx + j];//load 8 bases from batch1 sequence

				//--------------compute a tile of 8x8 cells-------------------
					for (k = 28; k >= 0; k -= 4) {
						uint32_t rbase = (rpac >> k) & 15;//get a base from batch1 sequence
						//-----load intermediate values--------------
						HD = global[ridx];
						h[0] = HD.x;
						e = HD.y;
						//-------------------------------------------
						int32_t prev_hm_diff = h[0] - _cudaGapOE;
#pragma unroll 8
						for (l = 28, m = 1; m < 9; l -= 4, m++) {
							uint32_t gbase = (gpac >> l) & 15;//get a base from batch2 sequence
							DEV_GET_SUB_SCORE(subScore, rbase, gbase);//check equality of rbase and gbase
							int32_t curr_hm_diff = h[m]- _cudaGapOE;
							f[m] = max(curr_hm_diff, f[m] - _cudaGapExtend);//whether to introduce or extend a gap in batch1 sequence
							h[m] = p[m] + subScore;//score if rbase is aligned to gbase
							h[m] = max(h[m], f[m]);
							h[m] = max(h[m], 0);
							e = max(prev_hm_diff, e - _cudaGapExtend);//whether to introduce or extend a gap in batch2 sequence
							prev_hm_diff = curr_hm_diff;
							h[m] = max(h[m], e);
							FIND_MAX(h[m], gidx + (m-1))//the current maximum score and corresponding end position on batch2 sequence
							p[m] = h[m-1];
						}
						//----------save intermediate values------------
						HD.x = h[m-1];
						HD.y = e;
						global[ridx] = HD;
						//---------------------------------------------
						maxXY_x = (prev_maxHH < maxHH) ? ridx : maxXY_x;//end position on batch1 sequence corresponding to current maximum score
						prev_maxHH = max(maxHH, prev_maxHH);
						ridx++;
					}
				//-------------------------------------------------------

			}

		}

		score[tid] = maxHH;//copy the max score to the output array in the GPU mem
		batch1_end[tid] = maxXY_x;//copy the end position on batch1 sequence to the output array in the GPU mem
		batch2_end[tid] = maxXY_y;//copy the end position on batch2 sequence to the output array in the GPU mem

		return;


}


__global__ void gasal_local_with_start_kernel(uint32_t *packed_batch1, uint32_t *packed_batch2, uint32_t *batch1_len, uint32_t *batch2_len, uint32_t *batch1_offset, uint32_t *batch2_offset, int32_t *score, int32_t *batch1_end, int32_t *batch2_end, int32_t *batch1_start, int32_t *batch2_start, int n_tasks) {
		int32_t i, j, k, m, l;
		int32_t e;
		int32_t maxHH = 0; //initialize the maximum score to zero
		int32_t prev_maxHH = 0;
		int32_t subScore;
		int32_t ridx, gidx;
		short2 HD;
		short2 initHD = make_short2(0, 0);
		int32_t maxXY_x = 0;
		int32_t maxXY_y = 0;
		const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;//thread ID
		if (tid >= n_tasks) return;
		uint32_t packed_batch2_idx = batch2_offset[tid] >> 3;//starting index of the batch2 sequence
		uint32_t packed_batch1_idx = batch1_offset[tid] >> 3;//starting index of the batch1 sequence
		uint32_t read_len = batch1_len[tid];
		uint32_t ref_len = batch2_len[tid];
		uint32_t batch1_regs = (read_len >> 3) + (read_len&7 ? 1 : 0);//number of 32-bit words holding batch1 sequence
		uint32_t batch2_regs = (ref_len >> 3) + (ref_len&7 ? 1 : 0);//number of 32-bit words holding batch2 sequence
		//-----arrays to save intermediate values------
		short2 global[MAX_SEQ_LEN];
		int32_t h[9];
		int32_t f[9];
		int32_t p[9];
		//---------------------------------------------
		for (i = 0; i < MAX_SEQ_LEN; i++) {
			global[i] = initHD;
		}


		for (i = 0; i < batch2_regs; i++) { //batch2 sequence in rows
			for (m = 0; m < 9; m++) {
					h[m] = 0;
					f[m] = 0;
					p[m] = 0;
			}
			register uint32_t gpac =packed_batch2[packed_batch2_idx + i];//load 8 packed bases from batch2 sequence
			gidx = i << 3;
			ridx = 0;
			for (j = 0; j < batch1_regs; j+=1) { //batch2 sequence in columns
				register uint32_t rpac =packed_batch1[packed_batch1_idx + j];//load 8 packed bases from batch1 sequence
					//-----------------compute a tile of 8x8 cells----------------------
					for (k = 28; k >= 0; k -= 4) {
						uint32_t rbase = (rpac >> k) & 15;// get a base from batch1 sequence
						//----------load intermediate values--------------
						HD = global[ridx];
						h[0] = HD.x;
						e = HD.y;
						//------------------------------------------------
#pragma unroll 8

						for (l = 28, m = 1; l >= 0; l -= 4, m++) {
							uint32_t gbase = (gpac >> l) & 15;//get a base from batch2 sequence
							DEV_GET_SUB_SCORE(subScore, rbase, gbase);//check equality of rbase and gbase
							f[m] = max(h[m]- _cudaGapOE, f[m] - _cudaGapExtend);//whether to introduce or extend a gap in batch1 sequence

							h[m] = p[m] + subScore;//score if gbase is aligned to rbase
							h[m] = max(h[m], f[m]);
							h[m] = max(h[m], 0);
							e = max(h[m-1] - _cudaGapOE, e - _cudaGapExtend);//whether to introduce or extend a gap in batch2 sequence
							h[m] = max(h[m], e);
							FIND_MAX(h[m], gidx + (m-1))//the current maximum score and corresponding end position on batch2 sequence
							p[m] = h[m-1];
						}
						//--------------save intermediate values-------------------
						HD.x = h[m-1];
						HD.y = e;
						global[ridx] = HD;
						//--------------------------------------------------------
						maxXY_x = (prev_maxHH < maxHH) ? ridx : maxXY_x;//end position on batch1 sequence corresponding to current maximum score
						prev_maxHH = max(maxHH, prev_maxHH);
						ridx++;
					}



			}

		}
		score[tid] = maxHH;//copy the max score to the output array in the GPU mem
		batch1_end[tid] = maxXY_x;//copy the end position on batch1 sequence to the output array in the GPU mem
		batch2_end[tid] = maxXY_y;//copy the end position on batch2 sequence to the output array in the GPU mem

		/*------------------Now to find the start position-----------------------*/

		int32_t rend_pos = maxXY_x;//end position on batch1 sequence
		int32_t gend_pos = maxXY_y;//end position on batch2 sequence
		int32_t fwd_score = maxHH;// the computed score


		int32_t rend_reg = ((rend_pos >> 3) + 1) < batch1_regs ? ((rend_pos >> 3) + 1) : batch1_regs;//the index of 32-bit word containing the end position on batch1 sequence
		int32_t gend_reg = ((gend_pos >> 3) + 1) < batch2_regs ? ((gend_pos >> 3) + 1) : batch2_regs;//the index of 32-bit word containing to end position on batch2 sequence


		packed_batch1_idx += (rend_reg - 1);
		packed_batch2_idx += (gend_reg - 1);


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
			register uint32_t gpac =packed_batch2[packed_batch2_idx - i];//load 8 packed bases from batch2 sequence
			gidx = gidx - 8;
			ridx = (rend_reg << 3) - 1;
			int32_t global_idx = 0;
			for (j = 0; j < rend_reg && maxHH < fwd_score; j+=1) {
				register uint32_t rpac =packed_batch1[packed_batch1_idx - j];//load 8 packed bases from batch1 sequence
				//--------------compute a tile of 8x8 cells-------------------
					for (k = 0; k <= 28 && maxHH < fwd_score; k += 4) {
						uint32_t rbase = (rpac >> k) & 15;//get a base from batch1 sequence
						//----------load intermediate values--------------
						HD = global[global_idx];
						h[0] = HD.x;
						e = HD.y;
						//-----------------------------------------------
	#pragma unroll 8
						for (l = 0, m = 1; l <= 28; l += 4, m++) {
							uint32_t gbase = (gpac >> l) & 15;//get a base from batch2 sequence
							DEV_GET_SUB_SCORE(subScore, rbase, gbase);//check equality of rbase and gbase
							f[m] = max(h[m]- _cudaGapOE, f[m] - _cudaGapExtend);//whether to introduce or extend a gap in batch1 sequence

							h[m] = p[m] + subScore;//score if gbase is aligned to rbase
							h[m] = max(h[m], f[m]);
							h[m] = max(h[m], 0);
							e = max(h[m-1] - _cudaGapOE, e - _cudaGapExtend);//whether to introduce or extend a gap in batch2 sequence
							h[m] = max(h[m], e);

							FIND_MAX(h[m], gidx - (m -1));//the current maximum score and corresponding start position on batch2 sequence
							p[m] = h[m-1];
						}
						//------------save intermediate values----------------
						HD.x = h[m-1];
						HD.y = e;
						global[global_idx] = HD;
						//----------------------------------------------------
						maxXY_x = (prev_maxHH < maxHH) ? ridx : maxXY_x;//start position on batch1 sequence corresponding to current maximum score
						prev_maxHH = max(maxHH, prev_maxHH);
						ridx--;
						global_idx++;
					}
					//-------------------------------------------------------



			}

		}
		//------------------------------------------------------------------------------------------------------------------------------------

		batch1_start[tid] = maxXY_x;//copy the start position on batch1 sequence to the output array in the GPU mem
		batch2_start[tid] = maxXY_y;//copy the start position on batch2 sequence to the output array in the GPU mem

		return;

}




__global__ void gasal_global_kernel(uint32_t *packed_batch1, uint32_t *packed_batch2,  uint32_t *batch1_len, uint32_t *batch2_len, uint32_t *batch1_offset, uint32_t *batch2_offset, int32_t *score, int n_tasks) {
		int32_t i, j, k, l, m;
		int32_t u = 0;
		int32_t e;
		int32_t maxHH =  MINUS_INF;//initialize the maximum score to -infinity
		int32_t subScore;

		int32_t ridx, gidx;
		short2 HD;

		const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;//thread ID
		if (tid >= n_tasks) return;
		uint32_t packed_batch2_idx = batch2_offset[tid] >> 3;//starting index of the batch2 sequence
		uint32_t packed_batch1_idx = batch1_offset[tid] >> 3;//starting index of the batch1 sequence
		uint32_t read_len = batch1_len[tid];
		uint32_t ref_len = batch2_len[tid];
		uint32_t batch1_regs = (read_len >> 3) + (read_len&7 ? 1 : 0);//number of 32-bit words holding sequence of batch1
		uint32_t batch2_regs = (ref_len >> 3) + (ref_len&7 ? 1 : 0);//number of 32-bit words holding sequence of batch2
		//-------arrays to save intermediate values----------------
		short2 global[MAX_SEQ_LEN];
		int32_t h[9];
		int32_t f[9];
		int32_t p[9];
		//----------------------------------------------------------
		global[0] = make_short2(0, MINUS_INF);
		for (i = 1; i < MAX_SEQ_LEN; i++) {
			global[i] = make_short2(-(_cudaGapO + (_cudaGapExtend*(i))), MINUS_INF);
		}


		h[u++] = 0;
		p[u++] = 0;
		for (i = 0; i < batch2_regs; i++) { //batch2 sequence in rows
			gidx = i << 3;
			ridx = 0;
			for (m = 1; m < 9; m++, u++) {
				h[m] = -(_cudaGapO + (_cudaGapExtend*(u-1)));
				f[m] = MINUS_INF;
				p[m] = -(_cudaGapO + (_cudaGapExtend*(u-1)));
			}
			register uint32_t gpac =packed_batch2[packed_batch2_idx + i];//load 8 packed bases from batch2 sequence


			for (j = 0; j < batch1_regs; /*++j*/ j+=1) { //batch1 sequence in columns

				register uint32_t rpac =packed_batch1[packed_batch1_idx + j];//load 8 packed bases from batch1 sequence
				//--------------compute a tile of 8x8 cells-------------------
					for (k = 28; k >= 0; k -= 4) {
						uint32_t rbase = (rpac >> k) & 15;//get a base from batch1 sequence
						//------------load intermediate values----------------------
						HD = global[ridx];
						h[0] = HD.x;
						e = HD.y;
						//----------------------------------------------------------
						int32_t prev_hm_diff = h[0] - _cudaGapOE;
	#pragma unroll 8
						for (l = 28, m = 1; m < 9; l -= 4, m++) {
							uint32_t gbase = (gpac >> l) & 15;//get a base from batch2 sequence
							DEV_GET_SUB_SCORE(subScore, rbase, gbase);//check the equality of rbase and gbase
							int32_t curr_hm_diff = h[m]- _cudaGapOE;
							f[m] = max(curr_hm_diff, f[m] - _cudaGapExtend);//whether to introduce or extend a gap in batch1 sequence
							h[m] = p[m] + subScore;//score if gbase is aligned to rbase
							h[m] = max(h[m], f[m]);
							e = max(prev_hm_diff, e - _cudaGapExtend);//whether to introduce or extend a gap in batch2 sequence
							prev_hm_diff = curr_hm_diff;
							h[m] = max(h[m], e);
							p[m] = h[m-1];

						}
						//--------------save intermediate values-------------------------
						HD.x = h[m-1];
						HD.y = e;//max(e, 0);
						global[ridx] = HD;
						ridx++;
						//--------------------------------------------------------------
						//------the last column of DP matrix------------
						if (ridx == read_len) {
							for (m = 1; m < 9; m++) {
								maxHH = ((gidx + (m -1)) == (ref_len - 1)) ? h[m] : maxHH;//if this is the last base of batch1 and batch2 sequence, then the max score is here
							}
						}
						//----------------------------------------------
					}
					//------------------------------------------------------------------




			}

		}
		score[tid] = maxHH;//copy the max score to the output array in the GPU mem

		return;


}

__global__ void gasal_semi_global_kernel(uint32_t *packed_batch1, uint32_t *packed_batch2,  uint32_t *batch1_len, uint32_t *batch2_len, uint32_t *batch1_offset, uint32_t *batch2_offset, int32_t *score, int32_t *batch2_end, int n_tasks) {
		int32_t i, j, k, l, m;
		int32_t e;
		int32_t maxHH =  MINUS_INF;//initialize the maximum score to -infinity
		int32_t subScore;
		int32_t ridx, gidx;
		short2 HD;
		short2 initHD = make_short2(0, 0);
		int32_t maxXY_y = 0;
		const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;//thread ID
		if (tid >= n_tasks) return;
		uint32_t packed_batch2_idx = batch2_offset[tid] >> 3;//starting index of the batch2 sequence
		uint32_t packed_batch1_idx = batch1_offset[tid] >> 3;//starting index of the batch1 sequence
		uint32_t read_len = batch1_len[tid];
		uint32_t ref_len = batch2_len[tid];
		uint32_t batch1_regs = (read_len >> 3) + (read_len&7 ? 1 : 0);//number of 32-bit words holding sequence of batch1
		uint32_t batch2_regs = (ref_len >> 3) + (ref_len&7 ? 1 : 0);//number of 32-bit words holding sequence of batch2
		//-------arrays to save intermediate values----------------
		short2 global[MAX_SEQ_LEN];
		int32_t h[9];
		int32_t f[9];
		int32_t p[9];
		//----------------------------------------------------------

		global[0] = make_short2(0, MINUS_INF);
		for (i = 1; i < MAX_SEQ_LEN; i++) {
			global[i] = make_short2(-(_cudaGapO + (_cudaGapExtend*(i))), MINUS_INF);
		}

		for (i = 0; i < batch2_regs; i++) { //batch2 sequence in rows
			gidx = i << 3;
			ridx = 0;
			for (m = 0; m < 9; m++) {
					h[m] = 0;
					f[m] = MINUS_INF;
					p[m] = 0;
			}

			register uint32_t gpac =packed_batch2[packed_batch2_idx + i];//load 8 packed bases from batch2 sequence

			for (j = 0; j < batch1_regs; j+=1) { //batch1 sequence in columns
				register uint32_t rpac =packed_batch1[packed_batch1_idx + j];//load 8 packed bases from batch1 sequence

				//--------------compute a tile of 8x8 cells-------------------
					for (k = 28; k >= 0; k -= 4) {
						uint32_t rbase = (rpac >> k) & 15;//get a base from batch1 sequence
						//------------load intermediate values----------------------
						HD = global[ridx];
						h[0] = HD.x;
						e = HD.y;
						//----------------------------------------------------------
						int32_t prev_hm_diff = h[0] - _cudaGapOE;
	#pragma unroll 8
						for (l = 28, m = 1; m < 9; l -= 4, m++) {
							uint32_t gbase = (gpac >> l) & 15;//get a base from batch2 sequence
							DEV_GET_SUB_SCORE(subScore, rbase, gbase);//check the equality of rbase and gbase
							int32_t curr_hm_diff = h[m]- _cudaGapOE;
							f[m] = max(curr_hm_diff, f[m] - _cudaGapExtend);//whether to introduce or extend a gap in batch1 sequence
							h[m] = p[m] + subScore;//score if gbase is aligned to rbase
							h[m] = max(h[m], f[m]);
							e = max(prev_hm_diff, e - _cudaGapExtend);//whether to introduce or extend a gap in batch2 sequence
							prev_hm_diff = curr_hm_diff;
							h[m] = max(h[m], e);
							p[m] = h[m-1];
						}
						//--------------save intermediate values-------------------------
						HD.x = h[m-1];
						HD.y = e;
						global[ridx] = HD;
						//---------------------------------------------------------------
						ridx++;
						//------the last column of DP matrix------------
						if (ridx == read_len) {
							//----find the maximum and the corresponding end position-----------
							for (m = 1; m < 9; m++) {
								maxXY_y = (h[m] > maxHH && (gidx + m -1) < ref_len) ? gidx + (m-1) : maxXY_y;
								maxHH = (h[m] > maxHH && (gidx + m -1) < ref_len) ? h[m] : maxHH;
							}
							//--------------------------------------------------------------------
						}
						//------------------------------------------------
					}
					//-------------------------------------------------------------

			}


		}
		score[tid] = maxHH;//copy the max score to the output array in the GPU mem
		batch2_end[tid] =  maxXY_y;//copy the end position on the batch2 sequence to the output array in the GPU mem

		return;


}

__global__ void gasal_semi_global_with_start_kernel(uint32_t *packed_batch1, uint32_t *packed_batch2, uint32_t *batch1_len, uint32_t *batch2_len, uint32_t *batch1_offset, uint32_t *batch2_offset, int32_t *score, int32_t *batch2_end, int32_t *batch2_start, int n_tasks) {

		int32_t i, j, k, l, m;
		int32_t e;
		int32_t maxHH =  MINUS_INF;//initialize the maximum score to -infinity
		int32_t subScore;
		int32_t ridx, gidx;
		short2 HD;
		short2 initHD = make_short2(0, 0);
		int32_t maxXY_y = 0;
		const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;//thread ID
		if (tid >= n_tasks) return;
		uint32_t packed_batch2_idx = batch2_offset[tid] >> 3;//starting index of the batch2 sequence
		uint32_t packed_batch1_idx = batch1_offset[tid] >> 3;//starting index of the batch1 sequence
		uint32_t read_len = batch1_len[tid];
		uint32_t ref_len = batch2_len[tid];
		uint32_t batch1_regs = (read_len >> 3) + (read_len&7 ? 1 : 0);//number of 32-bit words holding sequence of batch1
		uint32_t batch2_regs = (ref_len >> 3) + (ref_len&7 ? 1 : 0);//number of 32-bit words holding sequence of batch2
		//-------arrays to save intermediate values----------------
		short2 global[MAX_SEQ_LEN];
		int32_t h[9];
		int32_t f[9];
		int32_t p[9];
		//-------------------------------------------------------

		global[0] = make_short2(0, MINUS_INF);
		for (i = 1; i < MAX_SEQ_LEN; i++) {
			global[i] = make_short2(-(_cudaGapO + (_cudaGapExtend*(i))), MINUS_INF);
		}

		for (i = 0; i < batch2_regs; i++) { //batch2 sequence in rows
			gidx = i << 3;
			ridx = 0;
			for (m = 0; m < 9; m++) {
					h[m] = 0;
					f[m] = MINUS_INF;
					p[m] = 0;
			}

			register uint32_t gpac =packed_batch2[packed_batch2_idx + i];//load 8 packed bases from batch2 sequence

			for (j = 0; j < batch1_regs; /*++j*/ j+=1) { //batch1 sequence in columns
				register uint32_t rpac =packed_batch1[packed_batch1_idx + j];//load 8 packed bases from batch1 sequence

				//--------------compute a tile of 8x8 cells-------------------
					for (k = 28; k >= 0; k -= 4) {
						uint32_t rbase = (rpac >> k) & 15;//get a base from batch1 sequence
						//------------load intermediate values----------------------
						HD = global[ridx];
						h[0] = HD.x;
						e = HD.y;
						//----------------------------------------------------------
						int32_t prev_hm_diff = h[0] - _cudaGapOE;
	#pragma unroll 8
						for (l = 28, m = 1; m < 9; l -= 4, m++) {

							uint32_t gbase = (gpac >> l) & 15;//get a base from batch2 sequence
							DEV_GET_SUB_SCORE(subScore, rbase, gbase);//check the equality of rbase and gbase
							int32_t curr_hm_diff = h[m]- _cudaGapOE;
							f[m] = max(curr_hm_diff, f[m] - _cudaGapExtend);//whether to introduce or extend a gap in batch1 sequence
							h[m] = p[m] + subScore;//score if gbase is aligned to rbase
							h[m] = max(h[m], f[m]);
							e = max(prev_hm_diff, e - _cudaGapExtend);//whether to introduce or extend a gap in batch2 sequence
							prev_hm_diff = curr_hm_diff;
							h[m] = max(h[m], e);
							p[m] = h[m-1];
						}
						//--------------save intermediate values-------------------------
						HD.x = h[m-1];
						HD.y = e;
						global[ridx] = HD;
						//---------------------------------------------------------------
						ridx++;
						//------the last column of DP matrix------------
						if (ridx == read_len) {
							//----find the maximum and the corresponding end position-----------
							for (m = 1; m < 9; m++) {
								maxXY_y = (h[m] > maxHH && (gidx + m -1) < ref_len) ? gidx + (m-1) : maxXY_y;
								maxHH = (h[m] > maxHH && (gidx + m -1) < ref_len) ? h[m] : maxHH;
							}
							//------------------------------------------------------------------
						}
						//-----------------------------------------------
					}
					//------------------------------------------------------------




			}

		}

		score[tid] = maxHH;//copy the max score to the output array in the GPU mem
		batch2_end[tid] =  maxXY_y;//copy the end position on the batch2 sequence to the output array in the GPU mem


		/*------------------Now to find the start position-----------------------*/

		uint32_t reverse_batch1[(MAX_SEQ_LEN>>3)];//array to hold the reverse batch1 sequence
		uint32_t reverse_batch2[(MAX_SEQ_LEN>>3)];//array to hold the reverse batch1 sequence
		uint32_t reverse_batch1_reg;
		uint32_t reverse_batch2_reg;

		for (i = 0; i < (MAX_SEQ_LEN>>3); i++) {
			reverse_batch1[i] = 0;
		}
		for (i = 0; i < (MAX_SEQ_LEN>>3); i++) {
			reverse_batch2[i] = 0;
		}

		//--------reverse batch1 sequence--------------------
		for (i = read_len - 1, k = 0; i >= 0; i--, k++) {
			uint32_t orig_batch1_reg = i >> 3;
			uint32_t orig_symbol_pos = (((orig_batch1_reg + 1) << 3) - i) - 1;
			reverse_batch1_reg = k >> 3;
			uint32_t reverse_symbol_pos = (((reverse_batch1_reg + 1) << 3) - k) - 1;
			uint32_t orig_symbol = 0;
			orig_symbol = (packed_batch1[packed_batch1_idx + orig_batch1_reg] >> (orig_symbol_pos << 2)) & 15;
			reverse_batch1[reverse_batch1_reg] |= (orig_symbol << (reverse_symbol_pos << 2));
		}
		//---------------------------------------------------


		//--------reverse batch1 sequence--------------------
		for (i = ref_len - 1, k = 0; i >= 0; i--, k++) {
			uint32_t orig_batch2_reg = i >> 3;
			uint32_t orig_symbol_pos = (((orig_batch2_reg + 1) << 3) - i) - 1;
			reverse_batch2_reg = k >> 3;
			uint32_t reverse_symbol_pos = (((reverse_batch2_reg + 1) << 3) - k) - 1;
			uint32_t orig_symbol = 0;
			orig_symbol = (packed_batch2[packed_batch2_idx + orig_batch2_reg] >> (orig_symbol_pos << 2)) & 15;
			reverse_batch2[reverse_batch2_reg] |= (orig_symbol << (reverse_symbol_pos << 2));
		}
		//---------------------------------------------------

		int32_t gend_pos = maxXY_y;//end position on batch2 sequence
		int32_t fwd_score = maxHH;//the computed score

		//the index of 32-bit word containing the end position on batch2 sequence
		int32_t gend_reg = (batch2_regs - ((gend_pos >> 3) + 1)) > 0 ? (batch2_regs - ((gend_pos >> 3) + 1)) - 1 : (batch2_regs - ((gend_pos >> 3) + 1));

		maxHH = MINUS_INF;
		maxXY_y = 0;

		global[0] = make_short2(0, MINUS_INF);
		for (i = 1; i < MAX_SEQ_LEN; i++) {
			global[i] = make_short2(-(_cudaGapO + (_cudaGapExtend*(i))), MINUS_INF);
		}

		//------starting from the gend_reg, align the sequences in the reverse direction and exit if the max score >= fwd_score------
		for (i = gend_reg; i < batch2_regs && maxHH < fwd_score; i++) { //batch2 sequence in rows
			gidx = i << 3;
			ridx = 0;
			for (m = 0; m < 9; m++) {
				h[m] = 0;
				f[m] = MINUS_INF;
				p[m] = 0;
			}

			register uint32_t gpac =reverse_batch2[i];//load 8 packed bases from batch2 sequence

			for (j = 0; j < batch1_regs && maxHH < fwd_score;j+=1) { //batch1 sequence in columns
				register uint32_t rpac =reverse_batch1[j];//load 8 packed bases from batch2 sequence
				//--------------compute a tile of 8x8 cells-------------------
				for (k = 28; k >= 0; k -= 4) {
					uint32_t rbase = (rpac >> k) & 15;//get a base from batch1 sequence
					//------------load intermediate values----------------------
					HD = global[ridx];
					h[0] = HD.x;
					e = HD.y;
					//--------------------------------------------------------
					int32_t prev_hm_diff = h[0] - _cudaGapOE;
	#pragma unroll 8
					for (l = 28, m = 1; m < 9; l -= 4, m++) {
						uint32_t gbase = (gpac >> l) & 15;//get a base from batch2 sequence
						DEV_GET_SUB_SCORE(subScore, rbase, gbase);//check the equality of rbase and gbase
						int32_t curr_hm_diff = h[m]- _cudaGapOE;
						f[m] = max(curr_hm_diff, f[m] - _cudaGapExtend);//whether to introduce or extend a gap in batch1 sequence
						h[m] = p[m] + subScore;//score if gbase is aligned to rbase
						h[m] = max(h[m], f[m]);
						e = max(prev_hm_diff, e - _cudaGapExtend);//whether to introduce or extend a gap in batch2 sequence
						prev_hm_diff = curr_hm_diff;
						h[m] = max(h[m], e);
						p[m] = h[m-1];
					}
					//------------save intermediate values----------------------
					HD.x = h[m-1];
					HD.y = e;
					global[ridx] = HD;
					//----------------------------------------------------------
					ridx++;
					//------the last column of DP matrix------------
					if (ridx == read_len) {
						//----find the maximum and the corresponding end position-----------
						for (m = 1; m < 9; m++) {
							maxXY_y = (h[m] > maxHH && (gidx + (m -1)) < ref_len) ? gidx + (m-1) : maxXY_y;
							maxHH = (h[m] > maxHH && (gidx + (m -1)) < ref_len ) ? h[m] : maxHH;
						}
						//------------------------------------------------------------------
					}
					//----------------------------------------------
				}
				//---------------------------------------------------------------


			}

		}
		//-----------------------------------------------------------------------------------------------------------------

		batch2_start[tid] = (ref_len - 1) - maxXY_y;//copy the start position on batch2 sequence to the output array in the GPU mem

		return;



}
