


__global__ void gasal_pack_kernel_4bit(uint32_t* batch1,
		uint32_t* batch2, uint32_t *batch1_4bit, uint32_t* batch2_4bit,
		int batch1_tasks_per_thread, int batch2_tasks_per_thread, uint32_t total_batch1_regs, uint32_t total_batch2_regs) {

	int32_t i;
	const int32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x; // thbatch1 index.
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
#define N 0x4
#define DEV_GET_SUB_SCORE(score, rbase, gbase, dummy) \
      	score = (rbase == gbase) ?_cudaMatchScore : -_cudaMismatchScore;\
      	dummy = (rbase != 0) && (gbase != 0); \
      	score = dummy*score;

#define DEV_GET_SUB_SCORE_GLOBAL(score, rbase, gbase) \
      	score = (rbase == gbase) ?_cudaMatchScore : -_cudaMismatchScore;\
        score = ((rbase == N) || (gbase == N)) ? 0 : score;\

#define FIND_MAX(curr, gidx) \
								maxXY_y = (maxHH < curr) ? gidx : maxXY_y;\
								maxHH = (maxHH < curr) ? curr : maxHH;
//#define FIND_MAX(curr, gidx) \
//								maxXY_y = (curr < maxHH) ? maxXY_y : gidx;\
//								maxHH = (maxHH < curr) ? curr : maxHH;

__global__ void gasal_local_kernel(uint32_t *packed_batch1, uint32_t *packed_batch2,  uint32_t *batch1_len, uint32_t *batch2_len, uint32_t *batch1_offset, uint32_t *batch2_offset, int32_t *score, int32_t *batch1_end, int32_t *batch2_end, int n_tasks) {
	int32_t i, j, k, l;
	short m;
	int32_t e;
	int32_t maxHH = 0;
	int32_t prev_maxHH = 0;
	int32_t subScore;
	short ridx, gidx;
	short2 HD;
	short2 initHD = make_short2(0, 0);

	int32_t maxXY_y = 0;
	int32_t maxXY_x = 0;
	uint32_t packed_batch1_lmem[MAX_BATCH1_LEN>>3];
	uint32_t packed_batch2_lmem[MAX_BATCH2_LEN>>3];
	const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n_tasks) return;
	uint32_t packed_batch1_idx = batch1_offset[tid] >> 3;
	uint32_t packed_batch2_idx = batch2_offset[tid] >> 3;
	uint32_t packed_batch1_len = batch1_len[tid];
	uint32_t packed_batch2_len = batch2_len[tid];
	uint32_t packed_batch1_regs = (packed_batch1_len >> 3) + (packed_batch1_len&7 ? 1 : 0);//packed_batch1_regs[tid];
	uint32_t packed_batch2_regs = (packed_batch2_len >> 3) + (packed_batch2_len&7 ? 1 : 0);//packed_batch2_regs[tid];
	short2 global[MAX_SEQ_LEN];
	int32_t h[9];
	int32_t f[9];
	int32_t p[9];
	for (i = 0; i < packed_batch1_regs; i++) {
				packed_batch1_lmem[i] = packed_batch1[packed_batch1_idx + i];
		}
	for (i = 0; i < packed_batch2_regs; i++) {
				packed_batch2_lmem[i] = packed_batch2[packed_batch2_idx + i];
	}
	for (i = 0; i < MAX_SEQ_LEN; i++) {
		global[i] = initHD;
	}


	k = 0;
	for (i = 0; i < packed_batch2_regs - 1; i++) {
#pragma unroll
		for (m = 0; m < 9; m++) {
				h[m] = 0;
				f[m] = 0;
				p[m] = 0;
		}
		register uint32_t gpac =packed_batch2_lmem[i];

		gidx = i << 3;
		for (ridx = 0; ridx < packed_batch1_len;ridx++) {
			int curr_packed_batch1_reg = ridx >> 3;
			k = 28 - ((ridx & 7) << 2);
			register uint32_t rpac = packed_batch1_lmem[curr_packed_batch1_reg];
			uint32_t rbase = (rpac >> k) & 15;
			HD = global[ridx];
			h[0] = HD.x;
			p[0] = HD.x;
			e = HD.y;
#pragma unroll
			for (l = 28, m = 1; l >= 0; l -= 4, m++) {
				uint32_t gbase = (gpac >> l) & 15;
				DEV_GET_SUB_SCORE_GLOBAL(subScore, rbase, gbase);
				f[m] = max(h[m]- _cudaGapOE, f[m] - _cudaGapExtend);
				e = max(h[m-1] - _cudaGapOE, e - _cudaGapExtend);

				h[m] = p[m] + subScore;
				h[m] = max(h[m], f[m]);
				h[m] = max(h[m], e);

				h[m] = max(h[m], 0);
				FIND_MAX(h[m], gidx + (m-1))
				p[m] = h[m-1];
			}
			HD.x = h[m-1];
			HD.y = e;
			global[ridx] = HD;
			maxXY_x = (prev_maxHH < maxHH) ? ridx : maxXY_x;
			prev_maxHH = max(maxHH, prev_maxHH);
		}

	}

	gidx = i << 3;
	int packed_batch2_left = packed_batch2_len - gidx;
	register uint32_t gpac = packed_batch2_left > 0 ? packed_batch2_lmem[packed_batch2_regs - 1]: 0;
	for (m = 0; m < 9; m++) {
		h[m] = 0;
		f[m] = 0;
		p[m] = 0;
	}
	j = 0;
	for (k = 0; k < packed_batch1_len; k++){
		if (global[k].x + min(packed_batch2_left, k - (packed_batch1_len - 1)) <= maxHH) {
			j++;
		}
	}

	if (packed_batch2_left && (j < packed_batch1_len)){
		for (ridx = 0; ridx < packed_batch1_len; ridx++) {
			int curr_packed_batch1_reg = ridx >> 3;
			k = 28 - ((ridx & 7) << 2);
			register uint32_t rpac = packed_batch1_lmem[curr_packed_batch1_reg];
			uint32_t rbase = (rpac >> k) & 15;
			HD = global[ridx];
			h[0] = HD.x;
			p[0] = HD.x;
			e = HD.y;
#pragma unroll 8
			for (l = 28, m = 1; m <=packed_batch2_left; l -= 4, m++) {
				uint32_t gbase = (gpac >> l) & 15;
				DEV_GET_SUB_SCORE_GLOBAL(subScore, rbase, gbase);
				f[m] = max(h[m]- _cudaGapOE, f[m] - _cudaGapExtend);
				e = max(h[m-1] - _cudaGapOE, e - _cudaGapExtend);

				h[m] = p[m] + subScore;
				h[m] = max(h[m], f[m]);
				h[m] = max(h[m], e);

				h[m] = max(h[m], 0);
				FIND_MAX(h[m], gidx + (m-1))
				p[m] = h[m-1];
			}
			maxXY_x = (prev_maxHH < maxHH) ? ridx : maxXY_x;
			prev_maxHH = max(maxHH, prev_maxHH);
		}
	}


	score[tid] = maxHH;
	batch1_end[tid] = maxXY_x;
	batch2_end[tid] = maxXY_y;

	return;


}
__global__ void gasal_local_with_start_kernel(uint32_t *packed_batch1, uint32_t *packed_batch2, uint32_t *batch1_len, uint32_t *batch2_len, uint32_t *batch1_offset, uint32_t *batch2_offset, int32_t *score, int32_t *batch1_end, int32_t *batch2_end, int32_t *batch1_start, int32_t *batch2_start, int n_tasks) {
	int32_t i, j, k, l;
	short m;
	int32_t e;
	int32_t maxHH = 0;
	int32_t prev_maxHH = 0;
	int32_t subScore;

	short ridx, gidx;
	short2 HD;
	short2 initHD = make_short2(0, 0);

	int32_t maxXY_y = 0;
	int32_t maxXY_x = 0;
	uint32_t packed_batch1_lmem[MAX_BATCH1_LEN>>3];
	const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n_tasks) return;
	uint32_t packed_batch1_idx = batch1_offset[tid] >> 3;
	uint32_t packed_batch2_idx = batch2_offset[tid] >> 3;
	uint32_t packed_batch1_len = batch1_len[tid];
	uint32_t packed_batch2_len = batch2_len[tid];
	uint32_t packed_batch1_regs = (packed_batch1_len >> 3) + (packed_batch1_len&7 ? 1 : 0);//packed_batch1_regs[tid];
	uint32_t packed_batch2_regs = (packed_batch2_len >> 3) + (packed_batch2_len&7 ? 1 : 0);//packed_batch2_regs[tid];
	short2 global[MAX_SEQ_LEN];
	int32_t h[9];
	int32_t f[9];
	int32_t p[9];
	for (i = 0; i < packed_batch1_regs; i++) {
		packed_batch1_lmem[i] = __ldg(&(packed_batch1[packed_batch1_idx + i]));
	}

	for (i = 0; i < MAX_BATCH1_LEN; i++) {
		global[i] = initHD;
	}

	for (i = 0; i < packed_batch2_regs - 1; i++) { //genome in rows
#pragma unroll
		for (m = 0; m < 9; m++) {
			h[m] = 0;
			f[m] = 0;
			p[m] = 0;
		}
		register uint32_t gpac =packed_batch2[packed_batch2_idx + i];

		gidx = i << 3;

		for (ridx = 0; ridx < packed_batch1_len; /*++j*/ ridx++) {

			int curr_packed_batch1_reg = ridx >> 3;
			k = 28 - ((ridx & 7) << 2);
			register uint32_t rpac = packed_batch1_lmem[curr_packed_batch1_reg];
			uint32_t rbase = (rpac >> k) & 15;

			HD = global[ridx];
			h[0] = HD.x;
			p[0] = HD.x;
			e = HD.y;
#pragma unroll
			for (l = 28, m = 1; l >= 0; l -= 4, m++) {

				uint32_t gbase = (gpac >> l) & 15;
				DEV_GET_SUB_SCORE_GLOBAL(subScore, rbase, gbase);

				f[m] = max(h[m]- _cudaGapOE, f[m] - _cudaGapExtend);
				e = max(h[m-1] - _cudaGapOE, e - _cudaGapExtend);

				h[m] = p[m] + subScore;
				h[m] = max(h[m], f[m]);
				h[m] = max(h[m], e);


				h[m] = max(h[m], 0);
				FIND_MAX(h[m], gidx + (m-1))
				p[m] = h[m-1];
			}

			HD.x = h[m-1];
			HD.y = e;
			global[ridx] = HD;
			maxXY_x = (prev_maxHH < maxHH) ? ridx : maxXY_x;
			//maxXY_x = (maxHH < prev_maxHH) ? maxXY_x : ridx;
			prev_maxHH = max(maxHH, prev_maxHH);
		}
		//}




	}


	gidx = i << 3;
	int packed_batch2_left = packed_batch2_len - gidx;
	register uint32_t gpac = packed_batch2_left > 0 ? packed_batch2[packed_batch2_idx + (packed_batch2_regs - 1)]: 0;
	for (m = 0; m < 9; m++) {
		h[m] = 0;
		f[m] = 0;
		p[m] = 0;
	}
	j = 0;
	for (k = 0; k < packed_batch1_len; k++){
		if (global[k].x + min(packed_batch2_left, k - (packed_batch1_len - 1)) <= maxHH) {
			j++;
		}
	}
	if (packed_batch2_left && j < packed_batch1_len){
		for (ridx = 0; ridx < packed_batch1_len; ridx++) {

			int curr_packed_batch1_reg = ridx >> 3;
			k = 28 - ((ridx & 7) << 2);
			register uint32_t rpac = packed_batch1_lmem[curr_packed_batch1_reg];
			uint32_t rbase = (rpac >> k) & 15;
			HD = global[ridx];
			h[0] = HD.x;
			p[0] = HD.x;
			e = HD.y;
#pragma unroll 8
			for (l = 28, m = 1; m <=packed_batch2_left; l -= 4, m++) {

				uint32_t gbase = (gpac >> l) & 15;
				DEV_GET_SUB_SCORE_GLOBAL(subScore, rbase, gbase);

				f[m] = max(h[m]- _cudaGapOE, f[m] - _cudaGapExtend);
				e = max(h[m-1] - _cudaGapOE, e - _cudaGapExtend);

				h[m] = p[m] + subScore;
				h[m] = max(h[m], f[m]);
				h[m] = max(h[m], e);


				h[m] = max(h[m], 0);
				FIND_MAX(h[m], gidx + (m-1))
				p[m] = h[m-1];
			}

			maxXY_x = (prev_maxHH < maxHH) ? ridx : maxXY_x;
			//maxXY_x = (maxHH < prev_maxHH) ? maxXY_x : ridx;
			prev_maxHH = max(maxHH, prev_maxHH);
		}

	}


	batch1_end[tid] = maxXY_x;
	batch2_end[tid] = maxXY_y;

	int32_t fwd_score = maxHH;
	int32_t fwd_rend = maxXY_x;
	int32_t fwd_gend = maxXY_y;

	maxHH = 0;
	prev_maxHH = 0;
	maxXY_x = 0;
	maxXY_y = 0;



	for (i = 0; i < MAX_SEQ_LEN; i++) {
		global[i] = initHD;
	}

	for (i = 0; i < (packed_batch1_regs >> 1); i++) {
			uint32_t tmp = packed_batch1_lmem[i];
			packed_batch1_lmem[i] = packed_batch1_lmem[(packed_batch1_regs - 1) - i];
			packed_batch1_lmem[(packed_batch1_regs - 1) - i] = tmp;
	}

	int gend_reg = fwd_gend >> 3;

	packed_batch2_left = 8 - (((gend_reg + 1) << 3) - (fwd_gend + 1));
	gpac = packed_batch2_left > 0 ? packed_batch2[packed_batch2_idx + gend_reg]: 0;
	for (m = 0; m < 9; m++) {
		h[m] = 0;
		f[m] = 0;
		p[m] = 0;
	}
	gidx = fwd_gend;
	if (packed_batch2_left){
		for (ridx = fwd_rend; ridx >= 0; ridx--) {

			int curr_packed_batch1_reg = ridx >> 3;
			k = 28 - ((ridx & 7) << 2);
			register uint32_t rpac = packed_batch1_lmem[(packed_batch1_regs -1) - curr_packed_batch1_reg];
			uint32_t rbase = (rpac >> k) & 15;
			HD = global[fwd_rend - ridx];
			h[0] = HD.x;
			p[0] = HD.x;
			e = HD.y;
#pragma unroll 8
			for (l = 32 - (packed_batch2_left << 2), m = 1; m <=packed_batch2_left; l += 4, m++) {

				uint32_t gbase = (gpac >> l) & 15;
				DEV_GET_SUB_SCORE_GLOBAL(subScore, rbase, gbase);

				f[m] = max(h[m]- _cudaGapOE, f[m] - _cudaGapExtend);
				e = max(h[m-1] - _cudaGapOE, e - _cudaGapExtend);

				h[m] = p[m] + subScore;
				h[m] = max(h[m], f[m]);
				h[m] = max(h[m], e);


				h[m] = max(h[m], 0);
				FIND_MAX(h[m], gidx - (m-1))
				p[m] = h[m-1];
			}

			HD.x = h[m-1];
			HD.y = e;
			global[fwd_rend - ridx] = HD;
			maxXY_x = (prev_maxHH < maxHH) ? ridx : maxXY_x;
			//maxXY_x = (maxHH < prev_maxHH) ? maxXY_x : ridx;
			prev_maxHH = max(maxHH, prev_maxHH);
		}

	}
	gidx = (((gend_reg) << 3) + 8) - 1;
	for (i = (gend_reg - 1); i >= 0 && maxHH < fwd_score; i--) {
#pragma unroll
		for (m = 0; m < 9; m++) {
			h[m] = 0;
			f[m] = 0;
			p[m] = 0;
		}
		register uint32_t gpac =packed_batch2[packed_batch2_idx + i];

		gidx = gidx - 8;

		for (ridx = fwd_rend; ridx >=0 && maxHH < fwd_score; ridx--) {

			int curr_packed_batch1_reg = ridx >> 3;
			k = 28 - ((ridx & 7) << 2);
			register uint32_t rpac = packed_batch1_lmem[(packed_batch1_regs - 1) - curr_packed_batch1_reg];
			uint32_t rbase = (rpac >> k) & 15;

			HD = global[fwd_rend - ridx];
			h[0] = HD.x;
			p[0] = HD.x;
			e = HD.y;
#pragma unroll
			for (l = 0, m = 1; l <=28; l += 4, m++) {

				uint32_t gbase = (gpac >> l) & 15;
				DEV_GET_SUB_SCORE_GLOBAL(subScore, rbase, gbase);

				f[m] = max(h[m]- _cudaGapOE, f[m] - _cudaGapExtend);
				e = max(h[m-1] - _cudaGapOE, e - _cudaGapExtend);

				h[m] = p[m] + subScore;
				h[m] = max(h[m], f[m]);
				h[m] = max(h[m], e);


				h[m] = max(h[m], 0);
				FIND_MAX(h[m], gidx - (m-1))
				p[m] = h[m-1];
			}

			HD.x = h[m-1];
			HD.y = e;
			global[fwd_rend - ridx] = HD;
			maxXY_x = (prev_maxHH < maxHH) ? ridx : maxXY_x;
			//maxXY_x = (maxHH < prev_maxHH) ? maxXY_x : ridx;
			prev_maxHH = max(maxHH, prev_maxHH);
		}




	}

    score[tid] = maxHH;
    batch1_start[tid] = maxXY_x;
	batch2_start[tid] = maxXY_y;


	return;


}

__global__ void gasal_global_kernel(uint32_t *packed_batch1, uint32_t *packed_batch2,  uint32_t *batch1_len, uint32_t *batch2_len, uint32_t *batch1_offset, uint32_t *batch2_offset, int32_t *score, int n_tasks) {
	int32_t i, j, k, l, u =0;
	short m;
	int32_t e;
	int32_t maxHH = MINUS_INF;
	int32_t subScore;
	short ridx, gidx;
	short2 HD;

	uint32_t packed_batch1_lmem[MAX_BATCH1_LEN>>3];
	uint32_t packed_batch2_lmem[MAX_BATCH2_LEN>>3];
	const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n_tasks) return;
	uint32_t packed_batch1_idx = batch1_offset[tid] >> 3;
	uint32_t packed_batch2_idx = batch2_offset[tid] >> 3;
	uint32_t packed_batch1_len = batch1_len[tid];
	uint32_t packed_batch2_len = batch2_len[tid];
	uint32_t packed_batch1_regs = (packed_batch1_len >> 3) + (packed_batch1_len&7 ? 1 : 0);//packed_batch1_regs[tid];
	uint32_t packed_batch2_regs = (packed_batch2_len >> 3) + (packed_batch2_len&7 ? 1 : 0);//packed_batch2_regs[tid];
	short2 global[MAX_SEQ_LEN];
	int32_t h[9];
	int32_t f[9];
	int32_t p[9];
	for (i = 0; i < packed_batch1_regs; i++) {
				packed_batch1_lmem[i] = packed_batch1[packed_batch1_idx + i];
		}
	for (i = 0; i < packed_batch2_regs; i++) {
				packed_batch2_lmem[i] = packed_batch2[packed_batch2_idx + i];
	}
	for (i = 0; i < MAX_SEQ_LEN; i++) {
		global[i] = make_short2(-(_cudaGapO + (_cudaGapExtend*(i-1))), MINUS_INF);
	}


	k = 0;
	h[u++] = 0;
	p[u++] = 0;
	for (i = 0; i < packed_batch2_regs - 1; i++) {
#pragma unroll
		for (m = 0; m < 9; m++, u++) {
			h[m] = -(_cudaGapO + (_cudaGapExtend*(u-1)));
			f[m] = MINUS_INF;
			p[m] = -(_cudaGapO + (_cudaGapExtend*(u-1)));
		}
		register uint32_t gpac =packed_batch2_lmem[i];

		gidx = i << 3;
		for (ridx = 0; ridx < packed_batch1_len;ridx++) {
			int curr_packed_batch1_reg = ridx >> 3;
			k = 28 - ((ridx & 7) << 2);
			register uint32_t rpac = packed_batch1_lmem[curr_packed_batch1_reg];
			uint32_t rbase = (rpac >> k) & 15;
			HD = global[ridx];
			h[0] = HD.x;
			p[0] = HD.x;
			e = HD.y;
#pragma unroll
			for (l = 28, m = 1; l >= 0; l -= 4, m++) {
				uint32_t gbase = (gpac >> l) & 15;
				DEV_GET_SUB_SCORE_GLOBAL(subScore, rbase, gbase);
				f[m] = max(h[m]- _cudaGapOE, f[m] - _cudaGapExtend);
				e = max(h[m-1] - _cudaGapOE, e - _cudaGapExtend);

				h[m] = p[m] + subScore;
				h[m] = max(h[m], f[m]);
				h[m] = max(h[m], e);

				p[m] = h[m-1];
			}
			HD.x = h[m-1];
			HD.y = e;
			global[ridx] = HD;

		}

	}

	gidx = i << 3;
	int packed_batch2_left = packed_batch2_len - gidx;
	register uint32_t gpac = packed_batch2_left > 0 ? packed_batch2_lmem[packed_batch2_regs - 1]: 0;
	for (m = 0; m < 9; m++, u++) {
		h[m] = -(_cudaGapO + (_cudaGapExtend*(u-1)));
		f[m] = MINUS_INF;
		p[m] = -(_cudaGapO + (_cudaGapExtend*(u-1)));
	}
	j = 0;
	for (k = 0; k < packed_batch1_len; k++){
		if (global[k].x + min(packed_batch2_left, k - (packed_batch1_len - 1)) <= maxHH) {
			j++;
		}
	}

	if (packed_batch2_left && (j < packed_batch1_len)){
		for (ridx = 0; ridx < packed_batch1_len; ridx++) {
			int curr_packed_batch1_reg = ridx >> 3;
			k = 28 - ((ridx & 7) << 2);
			register uint32_t rpac = packed_batch1_lmem[curr_packed_batch1_reg];
			uint32_t rbase = (rpac >> k) & 15;
			HD = global[ridx];
			h[0] = HD.x;
			p[0] = HD.x;
			e = HD.y;
#pragma unroll 8
			for (l = 28, m = 1; m <=packed_batch2_left; l -= 4, m++) {
				uint32_t gbase = (gpac >> l) & 15;
				DEV_GET_SUB_SCORE_GLOBAL(subScore, rbase, gbase);
				f[m] = max(h[m]- _cudaGapOE, f[m] - _cudaGapExtend);
				e = max(h[m-1] - _cudaGapOE, e - _cudaGapExtend);

				h[m] = p[m] + subScore;
				h[m] = max(h[m], f[m]);
				h[m] = max(h[m], e);

				p[m] = h[m-1];
			}

		}

	}

	maxHH = packed_batch2_left ? h[packed_batch2_left] : h[8];
	score[tid] = maxHH;

	return;


}

__global__ void gasal_semi_global_kernel(uint32_t *packed_batch1, uint32_t *packed_batch2,  uint32_t *batch1_len, uint32_t *batch2_len, uint32_t *batch1_offset, uint32_t *batch2_offset, int32_t *score, int32_t *batch2_end, int n_tasks) {
	int32_t i, j, k, l;
	short m;
	int32_t e;
	int32_t maxHH = MINUS_INF;
	int32_t subScore;
	short ridx, gidx;
	short2 HD;

	int32_t maxXY_y = 0;
	uint32_t packed_batch1_lmem[MAX_BATCH1_LEN>>3];
	uint32_t packed_batch2_lmem[MAX_BATCH2_LEN>>3];
	const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n_tasks) return;
	uint32_t packed_batch1_idx = batch1_offset[tid] >> 3;
	uint32_t packed_batch2_idx = batch2_offset[tid] >> 3;
	uint32_t packed_batch1_len = batch1_len[tid];
	uint32_t packed_batch2_len = batch2_len[tid];
	uint32_t packed_batch1_regs = (packed_batch1_len >> 3) + (packed_batch1_len&7 ? 1 : 0);//packed_batch1_regs[tid];
	uint32_t packed_batch2_regs = (packed_batch2_len >> 3) + (packed_batch2_len&7 ? 1 : 0);//packed_batch2_regs[tid];
	short2 global[MAX_SEQ_LEN];
	int32_t h[9];
	int32_t f[9];
	int32_t p[9];
	for (i = 0; i < packed_batch1_regs; i++) {
				packed_batch1_lmem[i] = packed_batch1[packed_batch1_idx + i];
		}
	for (i = 0; i < packed_batch2_regs; i++) {
				packed_batch2_lmem[i] = packed_batch2[packed_batch2_idx + i];
	}
	for (i = 0; i < MAX_BATCH1_LEN; i++) {
		global[i] = make_short2(-(_cudaGapO + (_cudaGapExtend*(i))), MINUS_INF);
	}


	k = 0;
	for (i = 0; i < packed_batch2_regs - 1; i++) {
#pragma unroll
		for (m = 0; m < 9; m++) {
				h[m] = 0;
				f[m] = MINUS_INF;
				p[m] = 0;
		}
		register uint32_t gpac =packed_batch2_lmem[i];

		gidx = i << 3;
		for (ridx = 0; ridx < packed_batch1_len;ridx++) {
			int curr_packed_batch1_reg = ridx >> 3;
			k = 28 - ((ridx & 7) << 2);
			register uint32_t rpac = packed_batch1_lmem[curr_packed_batch1_reg];
			uint32_t rbase = (rpac >> k) & 15;
			HD = global[ridx];
			h[0] = HD.x;
			p[0] = HD.x;
			e = HD.y;
#pragma unroll
			for (l = 28, m = 1; l >= 0; l -= 4, m++) {
				uint32_t gbase = (gpac >> l) & 15;
				DEV_GET_SUB_SCORE_GLOBAL(subScore, rbase, gbase);
				f[m] = max(h[m]- _cudaGapOE, f[m] - _cudaGapExtend);
				e = max(h[m-1] - _cudaGapOE, e - _cudaGapExtend);

				h[m] = p[m] + subScore;
				h[m] = max(h[m], f[m]);
				h[m] = max(h[m], e);

				p[m] = h[m-1];
			}
			HD.x = h[m-1];
			HD.y = e;
			global[ridx] = HD;

		}
		for (m = 1; m < 9; m++) {
			maxXY_y = h[m] > maxHH ? gidx + (m-1) : maxXY_y;
			maxHH = h[m] > maxHH ? h[m] : maxHH;
		}

	}

	gidx = i << 3;
	int packed_batch2_left = packed_batch2_len - gidx;
	register uint32_t gpac = packed_batch2_left > 0 ? packed_batch2_lmem[packed_batch2_regs - 1]: 0;
	for (m = 0; m < 9; m++) {
		h[m] = 0;
		f[m] = MINUS_INF;
		p[m] = 0;
	}
	j = 0;
	for (k = 0; k < packed_batch1_len; k++){
		if (global[k].x + min(packed_batch2_left, k - (packed_batch1_len - 1)) <= maxHH) {
			j++;
		}
	}

	if (packed_batch2_left && (j < packed_batch1_len)){
		for (ridx = 0; ridx < packed_batch1_len; ridx++) {
			int curr_packed_batch1_reg = ridx >> 3;
			k = 28 - ((ridx & 7) << 2);
			register uint32_t rpac = packed_batch1_lmem[curr_packed_batch1_reg];
			uint32_t rbase = (rpac >> k) & 15;
			HD = global[ridx];
			h[0] = HD.x;
			p[0] = HD.x;
			e = HD.y;
#pragma unroll 8
			for (l = 28, m = 1; m <=packed_batch2_left; l -= 4, m++) {
				uint32_t gbase = (gpac >> l) & 15;
				DEV_GET_SUB_SCORE_GLOBAL(subScore, rbase, gbase);
				f[m] = max(h[m]- _cudaGapOE, f[m] - _cudaGapExtend);
				e = max(h[m-1] - _cudaGapOE, e - _cudaGapExtend);

				h[m] = p[m] + subScore;
				h[m] = max(h[m], f[m]);
				h[m] = max(h[m], e);

				p[m] = h[m-1];
			}

		}
		for (m = 1; m <= packed_batch2_left; m++) {
			maxXY_y = h[m] > maxHH ? gidx + (m-1) : maxXY_y;
			maxHH = h[m] > maxHH ? h[m] : maxHH;
		}

	}


	score[tid] = maxHH;
	batch2_end[tid] = maxXY_y;

	return;


}

__global__ void gasal_semi_global_with_start_kernel(uint32_t *packed_batch1, uint32_t *packed_batch2, uint32_t *batch1_len, uint32_t *batch2_len, uint32_t *batch1_offset, uint32_t *batch2_offset, int32_t *score, int32_t *batch2_end, int32_t *batch2_start, int n_tasks) {
	int32_t i, j, k, l;
	short m;
	int32_t e;
	int32_t maxHH = MINUS_INF;
	int32_t subScore;

	short ridx, gidx;
	short2 HD;

	int32_t maxXY_y = 0;
	uint32_t packed_batch1_lmem[MAX_BATCH1_LEN>>3];
	const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (tid >= n_tasks) return;
	uint32_t packed_batch1_idx = batch1_offset[tid] >> 3;
	uint32_t packed_batch2_idx = batch2_offset[tid] >> 3;
	uint32_t packed_batch1_len = batch1_len[tid];
	uint32_t packed_batch2_len = batch2_len[tid];
	uint32_t packed_batch1_regs = (packed_batch1_len >> 3) + (packed_batch1_len&7 ? 1 : 0);//packed_batch1_regs[tid];
	uint32_t packed_batch2_regs = (packed_batch2_len >> 3) + (packed_batch2_len&7 ? 1 : 0);//packed_batch2_regs[tid];
	short2 global[MAX_BATCH1_LEN];
	int32_t h[9];
	int32_t f[9];
	int32_t p[9];
	for (i = 0; i < packed_batch1_regs; i++) {
		packed_batch1_lmem[i] = __ldg(&(packed_batch1[packed_batch1_idx + i]));
	}

	for (i = 0; i < MAX_SEQ_LEN; i++) {
		global[i] =  make_short2(-(_cudaGapO + (_cudaGapExtend*(i))), MINUS_INF);
	}

	for (i = 0; i < packed_batch2_regs - 1; i++) { //genome in rows
#pragma unroll
		for (m = 0; m < 9; m++) {
			h[m] = 0;
			f[m] = MINUS_INF;
			p[m] = 0;
		}
		register uint32_t gpac =packed_batch2[packed_batch2_idx + i];

		gidx = i << 3;

		for (ridx = 0; ridx < packed_batch1_len; /*++j*/ ridx++) {

			int curr_packed_batch1_reg = ridx >> 3;
			k = 28 - ((ridx & 7) << 2);
			register uint32_t rpac = packed_batch1_lmem[curr_packed_batch1_reg];
			uint32_t rbase = (rpac >> k) & 15;

			HD = global[ridx];
			h[0] = HD.x;
			p[0] = HD.x;
			e = HD.y;
#pragma unroll
			for (l = 28, m = 1; l >= 0; l -= 4, m++) {

				uint32_t gbase = (gpac >> l) & 15;
				DEV_GET_SUB_SCORE_GLOBAL(subScore, rbase, gbase);

				f[m] = max(h[m]- _cudaGapOE, f[m] - _cudaGapExtend);
				e = max(h[m-1] - _cudaGapOE, e - _cudaGapExtend);

				h[m] = p[m] + subScore;
				h[m] = max(h[m], f[m]);
				h[m] = max(h[m], e);
				p[m] = h[m-1];
			}

			HD.x = h[m-1];
			HD.y = e;
			global[ridx] = HD;
		}
		//}
		for (m = 1; m < 9; m++) {
			maxXY_y = h[m] > maxHH ? gidx + (m-1) : maxXY_y;
			maxHH = h[m] > maxHH ? h[m] : maxHH;
		}



	}


	gidx = i << 3;
	int packed_batch2_left = packed_batch2_len - gidx;
	register uint32_t gpac = packed_batch2_left > 0 ? packed_batch2[packed_batch2_idx + (packed_batch2_regs - 1)]: 0;
	for (m = 0; m < 9; m++) {
		h[m] = 0;
		f[m] = MINUS_INF;
		p[m] = 0;
	}
	j = 0;
	for (k = 0; k < packed_batch1_len; k++){
		if (global[k].x + min(packed_batch2_left, k - (packed_batch1_len - 1)) <= maxHH) {
			j++;
		}
	}
	if (packed_batch2_left && j < packed_batch1_len){
		for (ridx = 0; ridx < packed_batch1_len; ridx++) {

			int curr_packed_batch1_reg = ridx >> 3;
			k = 28 - ((ridx & 7) << 2);
			register uint32_t rpac = packed_batch1_lmem[curr_packed_batch1_reg];
			uint32_t rbase = (rpac >> k) & 15;
			HD = global[ridx];
			h[0] = HD.x;
			p[0] = HD.x;
			e = HD.y;
#pragma unroll 8
			for (l = 28, m = 1; m <=packed_batch2_left; l -= 4, m++) {

				uint32_t gbase = (gpac >> l) & 15;
				DEV_GET_SUB_SCORE_GLOBAL(subScore, rbase, gbase);

				f[m] = max(h[m]- _cudaGapOE, f[m] - _cudaGapExtend);
				e = max(h[m-1] - _cudaGapOE, e - _cudaGapExtend);

				h[m] = p[m] + subScore;
				h[m] = max(h[m], f[m]);
				h[m] = max(h[m], e);


				p[m] = h[m-1];
			}

		}
		for (m = 1; m <= packed_batch2_left; m++) {
			maxXY_y = h[m] > maxHH ? gidx + (m-1) : maxXY_y;
			maxHH = h[m] > maxHH ? h[m] : maxHH;
		}


	}


	score[tid] = maxHH;

	batch2_end[tid] = maxXY_y;

	int32_t fwd_score = maxHH;
	int32_t fwd_gend = maxXY_y;

	maxHH = MINUS_INF;
	maxXY_y = 0;



	for (i = 0; i < MAX_SEQ_LEN; i++) {
		global[i] = make_short2(-(_cudaGapO + (_cudaGapExtend*(i))), MINUS_INF);
	}

	for (i = 0; i < (packed_batch1_regs >> 1); i++) {
			uint32_t tmp = packed_batch1_lmem[i];
			packed_batch1_lmem[i] = packed_batch1_lmem[(packed_batch1_regs - 1) - i];
			packed_batch1_lmem[(packed_batch1_regs - 1) - i] = tmp;
	}

	int gend_reg = fwd_gend >> 3;

	packed_batch2_left = 8 - (((gend_reg + 1) << 3) - (fwd_gend + 1));
	gpac = packed_batch2_left > 0 ? packed_batch2[packed_batch2_idx + gend_reg]: 0;
	for (m = 0; m < 9; m++) {
		h[m] = 0;
		f[m] = MINUS_INF;
		p[m] = 0;
	}
	gidx = fwd_gend;
	if (packed_batch2_left){
		for (ridx = packed_batch1_len - 1; ridx >= 0; ridx--) {

			int curr_packed_batch1_reg = ridx >> 3;
			k = 28 - ((ridx & 7) << 2);
			register uint32_t rpac = packed_batch1_lmem[(packed_batch1_regs -1) - curr_packed_batch1_reg];
			uint32_t rbase = (rpac >> k) & 15;
			HD = global[(packed_batch1_len - 1) - ridx];
			h[0] = HD.x;
			p[0] = HD.x;
			e = HD.y;
#pragma unroll 8
			for (l = 28 - (packed_batch2_left << 2), m = 1; m <=packed_batch2_left; l += 4, m++) {

				uint32_t gbase = (gpac >> l) & 15;
				DEV_GET_SUB_SCORE_GLOBAL(subScore, rbase, gbase);

				f[m] = max(h[m]- _cudaGapOE, f[m] - _cudaGapExtend);
				e = max(h[m-1] - _cudaGapOE, e - _cudaGapExtend);

				h[m] = p[m] + subScore;
				h[m] = max(h[m], f[m]);
				h[m] = max(h[m], e);

				p[m] = h[m-1];
			}

			HD.x = h[m-1];
			HD.y = e;
			global[(packed_batch1_len - 1) - ridx] = HD;
		}
		for (m = 1; m < 9; m++) {
			maxXY_y = h[m] > maxHH ? gidx - (m-1) : maxXY_y;
			maxHH = h[m] > maxHH ? h[m] : maxHH;
		}

	}
	gidx = (((gend_reg) << 3) + 8) - 1;
	for (i = (gend_reg - 1); i >= 0 && maxHH < fwd_score; i--) {
#pragma unroll
		for (m = 0; m < 9; m++) {
			h[m] = 0;
			f[m] = MINUS_INF;
			p[m] = 0;
		}
		register uint32_t gpac =packed_batch2[packed_batch2_idx + i];

		gidx = gidx - 8;

		for (ridx = packed_batch1_len - 1; ridx >=0 && maxHH < fwd_score; ridx--) {

			int curr_packed_batch1_reg = ridx >> 3;
			k = 28 - ((ridx & 7) << 2);
			register uint32_t rpac = packed_batch1_lmem[(packed_batch1_regs - 1) - curr_packed_batch1_reg];
			uint32_t rbase = (rpac >> k) & 15;

			HD = global[(packed_batch1_len - 1) - ridx];
			h[0] = HD.x;
			p[0] = HD.x;
			e = HD.y;
#pragma unroll
			for (l = 0, m = 1; l <=28; l += 4, m++) {

				uint32_t gbase = (gpac >> l) & 15;
				DEV_GET_SUB_SCORE_GLOBAL(subScore, rbase, gbase);

				f[m] = max(h[m]- _cudaGapOE, f[m] - _cudaGapExtend);
				e = max(h[m-1] - _cudaGapOE, e - _cudaGapExtend);

				h[m] = p[m] + subScore;
				h[m] = max(h[m], f[m]);
				h[m] = max(h[m], e);

				p[m] = h[m-1];
			}

			HD.x = h[m-1];
			HD.y = e;
			global[(packed_batch1_len - 1) - ridx] = HD;
		}
		for (m = 1; m < 9; m++) {
			maxXY_y = h[m] > maxHH ? gidx - (m-1) : maxXY_y;
			maxHH = h[m] > maxHH ? h[m] : maxHH;
		}

	}
	batch2_start[tid] = maxXY_y;


	return;


}
