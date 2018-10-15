#ifndef KERNEL_GLOBAL
#define KERNEL_GLOBAL

__global__ void gasal_global_kernel(uint32_t *packed_query_batch, uint32_t *packed_target_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, int32_t *score, int n_tasks) {
	int32_t i, j, k, l, m;
	int32_t u = 0;
	int32_t e;
	int32_t maxHH =  MINUS_INF;//initialize the maximum score to -infinity
	int32_t subScore;

	int32_t ridx, gidx;
	short2 HD;

	const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;//thread ID
	if (tid >= n_tasks) return;
	uint32_t packed_target_batch_idx = target_batch_offsets[tid] >> 3;//starting index of the target_batch sequence
	uint32_t packed_query_batch_idx = query_batch_offsets[tid] >> 3;//starting index of the query_batch sequence
	uint32_t read_len = query_batch_lens[tid];
	uint32_t ref_len = target_batch_lens[tid];
	uint32_t query_batch_regs = (read_len >> 3) + (read_len&7 ? 1 : 0);//number of 32-bit words holding sequence of query_batch
	uint32_t target_batch_regs = (ref_len >> 3) + (ref_len&7 ? 1 : 0);//number of 32-bit words holding sequence of target_batch
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
	for (i = 0; i < target_batch_regs; i++) { //target_batch sequence in rows, for all WORDS (i=WORD index)
		gidx = i << 3;
		ridx = 0;
		for (m = 1; m < 9; m++, u++) {
			h[m] = -(_cudaGapO + (_cudaGapExtend*(u-1))); 
			f[m] = MINUS_INF; 
			p[m] = -(_cudaGapO + (_cudaGapExtend*(u-1))); 
		}
		register uint32_t gpac =packed_target_batch[packed_target_batch_idx + i];//load 8 packed bases from target_batch sequence


		for (j = 0; j < query_batch_regs; /*++j*/ j+=1) { //query_batch sequence in columns, for all WORDS (j=WORD index).

			register uint32_t rpac =packed_query_batch[packed_query_batch_idx + j];//load 8 packed bases from query_batch sequence
			//--------------compute a tile of 8x8 cells-------------------
			for (k = 28; k >= 0; k -= 4) {
				uint32_t rbase = (rpac >> k) & 15;//get a base from query_batch sequence
				//------------load intermediate values----------------------
				HD = global[ridx];
				h[0] = HD.x;
				e = HD.y;
				//----------------------------------------------------------
				//int32_t prev_hm_diff = h[0] - _cudaGapOE;
	#pragma unroll 8
				for (l = 28, m = 1; m < 9; l -= 4, m++) {
					uint32_t gbase = (gpac >> l) & 15;//get a base from target_batch sequence
					DEV_GET_SUB_SCORE_GLOBAL(subScore, rbase, gbase);//check the equality of rbase and gbase
					//int32_t curr_hm_diff = h[m] - _cudaGapOE;
					f[m] = max(h[m]- _cudaGapOE, f[m] - _cudaGapExtend);//whether to introduce or extend a gap in query_batch sequence
					h[m] = p[m] + subScore;//score if gbase is aligned to rbase
					h[m] = max(h[m], f[m]);
					e = max(h[m - 1] - _cudaGapOE, e - _cudaGapExtend);//whether to introduce or extend a gap in target_batch sequence
					//prev_hm_diff=curr_hm_diff;
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
						maxHH = ((gidx + (m -1)) == (ref_len - 1)) ? h[m] : maxHH;//if this is the last base of query_batch and target_batch sequence, then the max score is here
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
#endif
