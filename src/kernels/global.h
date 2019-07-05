#ifndef __KERNEL_GLOBAL__
#define __KERNEL_GLOBAL__

#define CORE_GLOBAL_COMPUTE() \
		uint32_t gbase = (gpac >> l) & 15;\
		DEV_GET_SUB_SCORE_GLOBAL(subScore, rbase, gbase);\
		int32_t tmp_hm = p[m] + subScore;\
		h[m] = max(tmp_hm, f[m]);\
		h[m] = max(h[m], e);\
		f[m] =  (tmp_hm - _cudaGapOE) > (f[m] - _cudaGapExtend) ?  (tmp_hm - _cudaGapOE) : (f[m] - _cudaGapExtend);\
		e =  (tmp_hm - _cudaGapOE) > (e - _cudaGapExtend) ?  (tmp_hm - _cudaGapOE) : (e - _cudaGapExtend);\
		p[m] = h[m-1];\

#define CORE_GLOBAL_COMPUTE_TB(direction_reg) \
		uint32_t gbase = (gpac >> l) & 15;\
		DEV_GET_SUB_SCORE_GLOBAL(subScore, rbase, gbase);\
		int32_t tmp_hm = p[m] + subScore;\
		uint32_t m_or_x = tmp_hm >= p[m] ? 0 : 1;\
		h[m] = max(tmp_hm, f[m]);\
		h[m] = max(h[m], e);\
		direction_reg |= h[m] == tmp_hm ? m_or_x << (28 - ((m - 1) << 2)) : (h[m] == f[m] ? (uint32_t)3 << (28 - ((m - 1) << 2)) : (uint32_t)2 << (28 - ((m - 1) << 2)));\
		direction_reg |= (tmp_hm - _cudaGapOE) > (f[m] - _cudaGapExtend) ?  (uint32_t)0 : (uint32_t)1 << (31 - ((m - 1) << 2));\
		f[m] =  (tmp_hm - _cudaGapOE) > (f[m] - _cudaGapExtend) ?  (tmp_hm - _cudaGapOE) : (f[m] - _cudaGapExtend);\
		direction_reg|= (tmp_hm - _cudaGapOE) > (e - _cudaGapExtend) ?  (uint32_t)0 : (uint32_t)1 << (30 - ((m - 1) << 2));\
		e =  (tmp_hm - _cudaGapOE) > (e - _cudaGapExtend) ?  (tmp_hm - _cudaGapOE) : (e - _cudaGapExtend);\
		p[m] = h[m-1];\



template <typename S>
__global__ void gasal_global_kernel(uint32_t *packed_query_batch, uint32_t *packed_target_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, gasal_res_t *device_res, uint4 *packed_tb_matrices, int n_tasks)
{
	int32_t i, j, k, l, m;
	int32_t u = 0, r = 0;
	int32_t e;
	int32_t subScore;
	int tile_no = 0;

	int32_t ridx;
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
	short2 global[MAX_QUERY_LEN];
	int32_t h[9];
	int32_t f[9];
	int32_t p[9];
	int32_t max_h[9];
	//----------------------------------------------------------
	global[0] = make_short2(0, MINUS_INF);
	for (i = 1; i < MAX_QUERY_LEN; i++) {
		global[i] = make_short2(-(_cudaGapO + (_cudaGapExtend*(i))), MINUS_INF);
	}


	h[u++] = 0;
	p[r++] = 0;
	for (i = 0; i < target_batch_regs; i++) { //target_batch sequence in rows, for all WORDS (i=WORD index)
		ridx = 0;
		for (m = 1; m < 9; m++, u++, r++) {
			h[m] = -(_cudaGapO + (_cudaGapExtend*(u))); 
			f[m] = MINUS_INF; 
			p[m] = r == 1 ? 0 : -(_cudaGapO + (_cudaGapExtend*(r-1)));
		}
		register uint32_t gpac =packed_target_batch[packed_target_batch_idx + i];//load 8 packed bases from target_batch sequence


		for (j = 0; j < query_batch_regs; /*++j*/ j+=1) { //query_batch sequence in columns, for all WORDS (j=WORD index).

			register uint32_t rpac =packed_query_batch[packed_query_batch_idx + j];//load 8 packed bases from query_batch sequence

			//--------------compute a tile of 8x8 cells-------------------
			if (SAMETYPE(S, Int2Type<WITH_TB>)) {
				uint4 direction = make_uint4(0,0,0,0);
				uint32_t rbase = (rpac >> 28) & 15;//get a base from query_batch sequence
				//------------load intermediate values----------------------
				HD = global[ridx];
				h[0] = HD.x;
				e = HD.y;
#pragma unroll 8
				for (l = 28, m = 1; m < 9; l -= 4, m++) {
					CORE_GLOBAL_COMPUTE_TB(direction.x);
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
						max_h[m] = h[m];

					}
				}
				rbase = (rpac >> 24) & 15;//get a base from query_batch sequence
				//------------load intermediate values----------------------
				HD = global[ridx];
				h[0] = HD.x;
				e = HD.y;
#pragma unroll 8
				for (l = 28, m = 1; m < 9; l -= 4, m++) {
					CORE_GLOBAL_COMPUTE_TB(direction.y);
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
						max_h[m] = h[m];

					}
				}
				rbase = (rpac >> 20) & 15;//get a base from query_batch sequence
				//------------load intermediate values----------------------
				HD = global[ridx];
				h[0] = HD.x;
				e = HD.y;
#pragma unroll 8
				for (l = 28, m = 1; m < 9; l -= 4, m++) {
					CORE_GLOBAL_COMPUTE_TB(direction.z);
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
						max_h[m] = h[m];

					}
				}
				rbase = (rpac >> 16) & 15;//get a base from query_batch sequence
				//------------load intermediate values----------------------
				HD = global[ridx];
				h[0] = HD.x;
				e = HD.y;
#pragma unroll 8
				for (l = 28, m = 1; m < 9; l -= 4, m++) {
					CORE_GLOBAL_COMPUTE_TB(direction.w);
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
						max_h[m] = h[m];

					}
				}
				packed_tb_matrices[(tile_no*n_tasks) + tid] = direction;
				tile_no++;

				direction = make_uint4(0,0,0,0);
				rbase = (rpac >> 12) & 15;//get a base from query_batch sequence
				//------------load intermediate values----------------------
				HD = global[ridx];
				h[0] = HD.x;
				e = HD.y;
#pragma unroll 8
				for (l = 28, m = 1; m < 9; l -= 4, m++) {
					CORE_GLOBAL_COMPUTE_TB(direction.x);
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
						max_h[m] = h[m];

					}
				}
				rbase = (rpac >> 8) & 15;//get a base from query_batch sequence
				//------------load intermediate values----------------------
				HD = global[ridx];
				h[0] = HD.x;
				e = HD.y;
#pragma unroll 8
				for (l = 28, m = 1; m < 9; l -= 4, m++) {
					CORE_GLOBAL_COMPUTE_TB(direction.y);
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
						max_h[m] = h[m];

					}
				}
				rbase = (rpac >> 4) & 15;//get a base from query_batch sequence
				//------------load intermediate values----------------------
				HD = global[ridx];
				h[0] = HD.x;
				e = HD.y;
#pragma unroll 8
				for (l = 28, m = 1; m < 9; l -= 4, m++) {
					CORE_GLOBAL_COMPUTE_TB(direction.z);
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
						max_h[m] = h[m];

					}
				}
				rbase = rpac & 15;//get a base from query_batch sequence
				//------------load intermediate values----------------------
				HD = global[ridx];
				h[0] = HD.x;
				e = HD.y;
#pragma unroll 8
				for (l = 28, m = 1; m < 9; l -= 4, m++) {
					CORE_GLOBAL_COMPUTE_TB(direction.w);
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
						max_h[m] = h[m];

					}
				}
				packed_tb_matrices[(tile_no*n_tasks) + tid] = direction;
				tile_no++;

			}
			else{
			for (k = 28; k >= 0; k -= 4) {
				uint32_t rbase = (rpac >> k) & 15;//get a base from query_batch sequence
				//------------load intermediate values----------------------
				HD = global[ridx];
				h[0] = HD.x;
				e = HD.y;
				//----------------------------------------------------------
				#pragma unroll 8
				for (l = 28, m = 1; m < 9; l -= 4, m++) {
					CORE_GLOBAL_COMPUTE();
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
						max_h[m] = h[m];

					}
				}
				//----------------------------------------------
			}
		}
			//------------------------------------------------------------------
		}

	}
	
	device_res->aln_score[tid] = max_h[8 - ((target_batch_regs << 3) - (ref_len))];//copy the max score to the output array in the GPU mem

	return;

}
#endif
