//#define DEBUG

#ifdef DEBUG
#include <stdio.h>
#endif

#define A_PAK ('A'&0x0F)
#define C_PAK ('C'&0x0F)
#define G_PAK ('G'&0x0F)
#define T_PAK ('T'&0x0F)
#define N_PAK ('N'&0x0F)



__global__ void gasal_pack_kernel(uint32_t* unpacked_query_batch,
		uint32_t* unpacked_target_batch, uint32_t *packed_query_batch, uint32_t* packed_target_batch,
		int query_batch_tasks_per_thread, int target_batch_tasks_per_thread, uint32_t total_query_batch_regs, uint32_t total_target_batch_regs) {

	int32_t i;
	const int32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;//thread ID
	uint32_t n_threads = gridDim.x * blockDim.x;
	for (i = 0; i < query_batch_tasks_per_thread &&  (((i*n_threads)<<1) + (tid<<1) < total_query_batch_regs); ++i) {
		uint32_t *query_addr = &(unpacked_query_batch[(i*n_threads)<<1]);
		uint32_t reg1 = query_addr[(tid << 1)]; //load 4 bases of the query sequence from global memory
		uint32_t reg2 = query_addr[(tid << 1) + 1]; //load  another 4 bases
		uint32_t packed_reg = 0;
		packed_reg |= (reg1 & 15) << 28;        // ---
		packed_reg |= ((reg1 >> 8) & 15) << 24; //    |
		packed_reg |= ((reg1 >> 16) & 15) << 20;//    |
		packed_reg |= ((reg1 >> 24) & 15) << 16;//    |
		packed_reg |= (reg2 & 15) << 12;        //     > pack sequence
		packed_reg |= ((reg2 >> 8) & 15) << 8;  //    |
		packed_reg |= ((reg2 >> 16) & 15) << 4; //    |
		packed_reg |= ((reg2 >> 24) & 15);      //----
		uint32_t *packed_query_addr = &(packed_query_batch[i*n_threads]);
		packed_query_addr[tid] = packed_reg; //write 8 bases of packed query sequence to global memory
	}

	for (i = 0; i < target_batch_tasks_per_thread &&  (((i*n_threads)<<1) + (tid<<1)) < total_target_batch_regs; ++i) {
		uint32_t *target_addr = &(unpacked_target_batch[(i * n_threads)<<1]);
		uint32_t reg1 = target_addr[(tid << 1)]; //load 4 bases of the target sequence from global memory
		uint32_t reg2 = target_addr[(tid << 1) + 1]; //load  another 4 bases
		uint32_t packed_reg = 0;
		packed_reg |= (reg1 & 15) << 28;        // ---
		packed_reg |= ((reg1 >> 8) & 15) << 24; //    |
		packed_reg |= ((reg1 >> 16) & 15) << 20;//    |
		packed_reg |= ((reg1 >> 24) & 15) << 16;//    |
		packed_reg |= (reg2 & 15) << 12;        //     > pack sequence
		packed_reg |= ((reg2 >> 8) & 15) << 8;  //    |
		packed_reg |= ((reg2 >> 16) & 15) << 4; //    |
		packed_reg |= ((reg2 >> 24) & 15);      //----
		uint32_t *packed_target_addr = &(packed_target_batch[i * n_threads]);
		packed_target_addr[tid] = packed_reg; //write 8 bases of packed target sequence to global memory
	}

}

__global__ void	gasal_reversecomplement_kernel(uint32_t *packed_query_batch,uint32_t *packed_target_batch, uint32_t *query_batch_lens,
		uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, uint8_t *query_op, uint8_t *target_op, uint32_t  n_tasks){

	const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;//thread ID

	if (tid >= n_tasks) return;
	if (query_op[tid] == 0 && target_op[tid] == 0) return;		// if there's nothing to do (op=0, meaning sequence is Forward Natural), just exit the kernel ASAP. 


	uint32_t packed_target_batch_idx = target_batch_offsets[tid] >> 3;//starting index of the target_batch sequence
	uint32_t packed_query_batch_idx = query_batch_offsets[tid] >> 3;//starting index of the query_batch sequence
	uint32_t read_len = query_batch_lens[tid];
	uint32_t ref_len = target_batch_lens[tid];
	uint32_t query_batch_regs = (read_len >> 3) + (read_len&7 ? 1 : 0);//number of 32-bit words holding sequence of query_batch
	uint32_t target_batch_regs = (ref_len >> 3) + (ref_len&7 ? 1 : 0);//number of 32-bit words holding sequence of target_batch

	uint32_t query_batch_regs_to_swap = (query_batch_regs >> 1) + (query_batch_regs & 1); // that's (query_batch_regs / 2) + 1 if it's odd, + 0 otherwise. Used for reverse (we start a both ends, and finish at the center of the sequence)
	uint32_t target_batch_regs_to_swap = (target_batch_regs >> 1) + (target_batch_regs & 1); // that's (target_batch_regs / 2) + 1 if it's odd, + 0 otherwise. Used for reverse (we start a both ends, and finish at the center of the sequence)


	// variables used dependent on target and query: 

	uint8_t *op = NULL;
	uint32_t *packed_batch = NULL;
	uint32_t *batch_regs = NULL;
	uint32_t *batch_regs_to_swap = NULL;
	uint32_t *packed_batch_idx = NULL;

	// avoid useless code duplicate thanks to pointers to route the data flow where it should be, twice.
	#pragma unroll 2
	for (int p = 0; p < 2; p++)
	{
		switch(p)
		{
			case QUERY:
				op = query_op;
				packed_batch = packed_query_batch;
				batch_regs = &query_batch_regs;
				batch_regs_to_swap = &query_batch_regs_to_swap;
				packed_batch_idx = &packed_query_batch_idx;
				break;
			case TARGET:
				op = target_op;
				packed_batch = packed_target_batch;
				batch_regs = &target_batch_regs;
				batch_regs_to_swap = &target_batch_regs_to_swap;
				packed_batch_idx = &packed_target_batch_idx;
				break;
			default:
				break;
		}

		if (*(op + tid) & 0x01) // reverse
		{
			// deal with N's : read last word, find how many N's, store that number as offset, and pad with that many for the last 
			uint8_t nbr_N = 0;
			for (int j = 0; j < 32; j = j + 4)
			{
				nbr_N += (((*(packed_batch + *(packed_batch_idx) + *(batch_regs)-1) & (0x0F << j)) >> j) == N_PAK);
			}
	#ifdef DEBUG
			//printf("KERNEL_DEBUG: nbr_N=%d\n", nbr_N);
	#endif
			nbr_N = nbr_N << 2; // we operate on nibbles so we will need to do our shifts 4 bits by 4 bits, so 4*nbr_N

			for (uint32_t i = 0; i < *(batch_regs_to_swap); i++) // reverse all words. There's a catch with the last word (in the middle of the sequence), see final if.
			{
				uint32_t rpac_1 = *(packed_batch + *(packed_batch_idx) + i); //load 8 packed bases from head
				uint32_t rpac_2 = ((*(packed_batch + *(packed_batch_idx) + *(batch_regs)-2 - i)) << (32-nbr_N)) | ((*(packed_batch + *(packed_batch_idx) + *(batch_regs)-1 - i)) >> nbr_N);


				uint32_t reverse_rpac_1 = 0;
				uint32_t reverse_rpac_2 = 0;


	#pragma unroll 8
				for(int k = 28; k >= 0; k = k - 4)		// reverse 32-bits word... is pragma-unrolled. 
				{
					reverse_rpac_1 |= ((rpac_1 & (0x0F << k)) >> (k)) << (28-k);
					reverse_rpac_2 |= ((rpac_2 & (0x0F << k)) >> (k)) << (28-k);
				}
				// last swap operated manually, because of its irregular size (32 - 4*nbr_N bits, hence 8 - nbr_N nibbles)


				uint32_t to_queue_1 = (reverse_rpac_1 << nbr_N) | ((*(packed_batch + *(packed_batch_idx) + *(batch_regs)-1 - i)) & ((1<<nbr_N) - 1));
				uint32_t to_queue_2 = ((*(packed_batch + *(packed_batch_idx) + *(batch_regs)-2 - i)) & (0xFFFFFFFF - ((1<<nbr_N) - 1))) | (reverse_rpac_1 >> (32-nbr_N));

	#ifdef DEBUG				
				//printf("KERNEL DEBUG: rpac_1 Word before reverse: %x, after: %x, split into %x + %x \n", rpac_1, reverse_rpac_1, to_queue_2, to_queue_1 );
				//printf("KERNEL DEBUG: rpac_2 Word before reverse: %x, after: %x\n", rpac_2, reverse_rpac_2 );
	#endif

				*(packed_batch + *(packed_batch_idx) + i) = reverse_rpac_2;
				(*(packed_batch + *(packed_batch_idx) + *(batch_regs)-1 - i)) = to_queue_1;
				if (i!=*(batch_regs_to_swap)-1)
					(*(packed_batch + *(packed_batch_idx) + *(batch_regs)-2 - i)) = to_queue_2;


			} // end for
		} // end if(reverse)

		if (*(op+tid) & 0x02) // complement
		{
			for (uint32_t i = 0; i < *(batch_regs); i++) // reverse all words. There's a catch with the last word (in the middle of the sequence), see final if.
			{
				uint32_t rpac = *(packed_batch + *(packed_batch_idx) + i); //load 8 packed bases from head
				uint32_t nucleotide = 0;

	#pragma unroll 8
				for(int k = 28; k >= 0; k = k - 4)		// complement 32-bits word... is pragma-unrolled. 
				{
					nucleotide = (rpac & (0x0F << k)) >> (k);
					switch(nucleotide)
					{
						case A_PAK:
							nucleotide = T_PAK;
							break;
						case C_PAK:
							nucleotide = G_PAK;
							break;
						case T_PAK:
							nucleotide = A_PAK;
							break;
						case G_PAK:
							nucleotide = C_PAK;
							break;
						default:
							break;
					}
					rpac = (rpac & (0xFFFFFFFF - (0x0F << k))) | nucleotide << k;
				}

	#ifdef DEBUG
				//printf("KERNEL DEBUG: Word read : %x, after complement: %x\n", *(packed_batch + *(packed_batch_idx) + i), rpac);
	#endif

				*(packed_batch + *(packed_batch_idx) + i) = rpac;

			} // end for
		} // end if(complement)



	}

	return;
}


__constant__ int32_t _cudaGapO; /*gap open penalty*/
__constant__ int32_t _cudaGapOE; /*sum of gap open and extension penalties*/
__constant__ int32_t _cudaGapExtend; /*sum of gap extend*/
__constant__ int32_t _cudaMatchScore; /*score for a match*/
__constant__ int32_t _cudaMismatchScore; /*penalty for a mismatch*/

#define MINUS_INF SHRT_MIN

#define N_VALUE (N_CODE & 0xF)

#ifdef N_PENALTY
#define DEV_GET_SUB_SCORE_LOCAL(score, rbase, gbase) \
	score = (rbase == gbase) ?_cudaMatchScore : -_cudaMismatchScore;\
score = ((rbase == N_VALUE) || (gbase == N_VALUE)) ? -N_PENALTY : score;\

#define DEV_GET_SUB_SCORE_GLOBAL(score, rbase, gbase) \
	score = (rbase == gbase) ?_cudaMatchScore : -_cudaMismatchScore;\
score = ((rbase == N_VALUE) || (gbase == N_VALUE)) ? -N_PENALTY : score;\

#else
#define DEV_GET_SUB_SCORE_LOCAL(score, rbase, gbase) \
	score = (rbase == gbase) ?_cudaMatchScore : -_cudaMismatchScore;\
score = ((rbase == N_VALUE) || (gbase == N_VALUE)) ? 0 : score;\

#define DEV_GET_SUB_SCORE_GLOBAL(score, rbase, gbase) \
	score = (rbase == gbase) ?_cudaMatchScore : -_cudaMismatchScore;\

#endif

#define MAX(a,b) (a>b?a:b)
#define MIN(a,b) (a>b?b:a)


#define FIND_MAX(curr, gidx) \
	maxXY_y = (maxHH < curr) ? gidx : maxXY_y;\
maxHH = (maxHH < curr) ? curr : maxHH;


__global__ void gasal_local_kernel(uint32_t *packed_query_batch, uint32_t *packed_target_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, int32_t *score, int32_t *query_batch_end, int32_t *target_batch_end, int n_tasks) {
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
		for (j = 0; j < query_batch_regs; j+=1) { //query_batch sequence in columns
			register uint32_t rpac =packed_query_batch[packed_query_batch_idx + j];//load 8 bases from query_batch sequence

			//--------------compute a tile of 8x8 cells-------------------
			for (k = 28; k >= 0; k -= 4) {
				uint32_t rbase = (rpac >> k) & 15;//get a base from query_batch sequence
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

		}

	}

	score[tid] = maxHH;//copy the max score to the output array in the GPU mem
	query_batch_end[tid] = maxXY_x;//copy the end position on query_batch sequence to the output array in the GPU mem
	target_batch_end[tid] = maxXY_y;//copy the end position on target_batch sequence to the output array in the GPU mem

	return;


}


__global__ void gasal_local_with_start_kernel(uint32_t *packed_query_batch, uint32_t *packed_target_batch, uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, int32_t *score, int32_t *query_batch_end, int32_t *target_batch_end, int32_t *query_batch_start, int32_t *target_batch_start, int n_tasks) {
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
	uint32_t packed_target_batch_idx = target_batch_offsets[tid] >> 3;//starting index of the target_batch sequence
	uint32_t packed_query_batch_idx = query_batch_offsets[tid] >> 3;//starting index of the query_batch sequence
	uint32_t read_len = query_batch_lens[tid];
	uint32_t ref_len = target_batch_lens[tid];
	uint32_t query_batch_regs = (read_len >> 3) + (read_len&7 ? 1 : 0);//number of 32-bit words holding query_batch sequence
	uint32_t target_batch_regs = (ref_len >> 3) + (ref_len&7 ? 1 : 0);//number of 32-bit words holding target_batch sequence
	//-----arrays to save intermediate values------
	short2 global[MAX_SEQ_LEN];
	int32_t h[9];
	int32_t f[9];
	int32_t p[9];
	//---------------------------------------------
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
		for (j = 0; j < query_batch_regs; j+=1) { //target_batch sequence in columns
			register uint32_t rpac =packed_query_batch[packed_query_batch_idx + j];//load 8 packed bases from query_batch sequence
			//-----------------compute a tile of 8x8 cells----------------------
			for (k = 28; k >= 0; k -= 4) {
				uint32_t rbase = (rpac >> k) & 15;// get a base from query_batch sequence
				//----------load intermediate values--------------
				HD = global[ridx];
				h[0] = HD.x;
				e = HD.y;
				//------------------------------------------------
	#pragma unroll 8

				for (l = 28, m = 1; l >= 0; l -= 4, m++) {
					uint32_t gbase = (gpac >> l) & 15;//get a base from target_batch sequence
					DEV_GET_SUB_SCORE_LOCAL(subScore, rbase, gbase);//check equality of rbase and gbase
					f[m] = max(h[m]- _cudaGapOE, f[m] - _cudaGapExtend);//whether to introduce or extend a gap in query_batch sequence

					h[m] = p[m] + subScore;//score if gbase is aligned to rbase
					h[m] = max(h[m], f[m]);
					h[m] = max(h[m], 0);
					e = max(h[m-1] - _cudaGapOE, e - _cudaGapExtend);//whether to introduce or extend a gap in target_batch sequence
					h[m] = max(h[m], e);
					FIND_MAX(h[m], gidx + (m-1))//the current maximum score and corresponding end position on target_batch sequence
						p[m] = h[m-1];
				}
				//--------------save intermediate values-------------------
				HD.x = h[m-1];
				HD.y = e;
				global[ridx] = HD;
				//--------------------------------------------------------
				maxXY_x = (prev_maxHH < maxHH) ? ridx : maxXY_x;//end position on query_batch sequence corresponding to current maximum score
				prev_maxHH = max(maxHH, prev_maxHH);
				ridx++;
			}



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

__global__ void gasal_semi_global_kernel(uint32_t *packed_query_batch, uint32_t *packed_target_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, int32_t *score, int32_t *target_batch_end, int n_tasks) {
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

	for (i = 0; i < target_batch_regs; i++) { //target_batch sequence in rows
		gidx = i << 3;
		ridx = 0;
		for (m = 0; m < 9; m++) {
			h[m] = 0;
			f[m] = MINUS_INF;
			p[m] = 0;
		}

		register uint32_t gpac =packed_target_batch[packed_target_batch_idx + i];//load 8 packed bases from target_batch sequence

		for (j = 0; j < query_batch_regs; j+=1) { //query_batch sequence in columns
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
	target_batch_end[tid] =  maxXY_y;//copy the end position on the target_batch sequence to the output array in the GPU mem

	return;


}

__global__ void gasal_semi_global_with_start_kernel(uint32_t *packed_query_batch, uint32_t *packed_target_batch, uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, int32_t *score, int32_t *target_batch_end, int32_t *target_batch_start, int n_tasks) {

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
	//-------------------------------------------------------

	global[0] = make_short2(0, MINUS_INF);
	for (i = 1; i < MAX_SEQ_LEN; i++) {
		global[i] = make_short2(-(_cudaGapO + (_cudaGapExtend*(i))), MINUS_INF);
	}

	for (i = 0; i < target_batch_regs; i++) { //target_batch sequence in rows
		gidx = i << 3;
		ridx = 0;
		for (m = 0; m < 9; m++) {
			h[m] = 0;
			f[m] = MINUS_INF;
			p[m] = 0;
		}

		register uint32_t gpac =packed_target_batch[packed_target_batch_idx + i];//load 8 packed bases from target_batch sequence

		for (j = 0; j < query_batch_regs; /*++j*/ j+=1) { //query_batch sequence in columns
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
	target_batch_end[tid] =  maxXY_y;//copy the end position on the target_batch sequence to the output array in the GPU mem


	/*------------------Now to find the start position-----------------------*/

	uint32_t reverse_query_batch[(MAX_SEQ_LEN>>3)];//array to hold the reverse query_batch sequence
	uint32_t reverse_target_batch[(MAX_SEQ_LEN>>3)];//array to hold the reverse query_batch sequence
	uint32_t reverse_query_batch_reg;
	uint32_t reverse_target_batch_reg;

	for (i = 0; i < (MAX_SEQ_LEN>>3); i++) {
		reverse_query_batch[i] = 0;
	}
	for (i = 0; i < (MAX_SEQ_LEN>>3); i++) {
		reverse_target_batch[i] = 0;
	}

	//--------reverse query_batch sequence--------------------
	for (i = read_len - 1, k = 0; i >= 0; i--, k++) {
		uint32_t orig_query_batch_reg = i >> 3;
		uint32_t orig_symbol_pos = (((orig_query_batch_reg + 1) << 3) - i) - 1;
		reverse_query_batch_reg = k >> 3;
		uint32_t reverse_symbol_pos = (((reverse_query_batch_reg + 1) << 3) - k) - 1;
		uint32_t orig_symbol = 0;
		orig_symbol = (packed_query_batch[packed_query_batch_idx + orig_query_batch_reg] >> (orig_symbol_pos << 2)) & 15;
		reverse_query_batch[reverse_query_batch_reg] |= (orig_symbol << (reverse_symbol_pos << 2));
	}
	//---------------------------------------------------


	//--------reverse query_batch sequence--------------------
	for (i = ref_len - 1, k = 0; i >= 0; i--, k++) {
		uint32_t orig_target_batch_reg = i >> 3;
		uint32_t orig_symbol_pos = (((orig_target_batch_reg + 1) << 3) - i) - 1;
		reverse_target_batch_reg = k >> 3;
		uint32_t reverse_symbol_pos = (((reverse_target_batch_reg + 1) << 3) - k) - 1;
		uint32_t orig_symbol = 0;
		orig_symbol = (packed_target_batch[packed_target_batch_idx + orig_target_batch_reg] >> (orig_symbol_pos << 2)) & 15;
		reverse_target_batch[reverse_target_batch_reg] |= (orig_symbol << (reverse_symbol_pos << 2));
	}
	//---------------------------------------------------

	int32_t gend_pos = maxXY_y;//end position on target_batch sequence
	int32_t fwd_score = maxHH;//the computed score

	//the index of 32-bit word containing the end position on target_batch sequence
	int32_t gend_reg = (target_batch_regs - ((gend_pos >> 3) + 1)) > 0 ? (target_batch_regs - ((gend_pos >> 3) + 1)) - 1 : (target_batch_regs - ((gend_pos >> 3) + 1));

	maxHH = MINUS_INF;
	maxXY_y = 0;

	global[0] = make_short2(0, MINUS_INF);
	for (i = 1; i < MAX_SEQ_LEN; i++) {
		global[i] = make_short2(-(_cudaGapO + (_cudaGapExtend*(i))), MINUS_INF);
	}

	//------starting from the gend_reg, align the sequences in the reverse direction and exit if the max score >= fwd_score------
	for (i = gend_reg; i < target_batch_regs && maxHH < fwd_score; i++) { //target_batch sequence in rows
		gidx = i << 3;
		ridx = 0;
		for (m = 0; m < 9; m++) {
			h[m] = 0;
			f[m] = MINUS_INF;
			p[m] = 0;
		}

		register uint32_t gpac =reverse_target_batch[i];//load 8 packed bases from target_batch sequence

		for (j = 0; j < query_batch_regs && maxHH < fwd_score;j+=1) { //query_batch sequence in columns
			register uint32_t rpac =reverse_query_batch[j];//load 8 packed bases from target_batch sequence
			//--------------compute a tile of 8x8 cells-------------------
			for (k = 28; k >= 0; k -= 4) {
				uint32_t rbase = (rpac >> k) & 15;//get a base from query_batch sequence
				//------------load intermediate values----------------------
				HD = global[ridx];
				h[0] = HD.x;
				e = HD.y;
				//--------------------------------------------------------
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

	target_batch_start[tid] = (ref_len - 1) - maxXY_y;//copy the start position on target_batch sequence to the output array in the GPU mem

	return;



}

__global__ void gasal_banded_kernel(uint32_t *packed_query_batch, uint32_t *packed_target_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, int32_t *score, int32_t *query_batch_end, int32_t *target_batch_end, int n_tasks, int32_t k_band_width) {
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
	int32_t k_band_width_loc = k_band_width<<3;
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
						#ifdef DEBUG
						if(tid==0)
						{
							printf("(%d) - ",x_minus_y);
						}
						#endif

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
			#ifdef DEBUG
			if(tid==0)
			printf("\n");
			#endif
		}

	}

	score[tid] = maxHH;//copy the max score to the output array in the GPU mem
	query_batch_end[tid] = maxXY_x;//copy the end position on query_batch sequence to the output array in the GPU mem
	target_batch_end[tid] = maxXY_y;//copy the end position on target_batch sequence to the output array in the GPU mem

	return;

}

__global__ void gasal_banded_with_start_kernel(uint32_t *packed_query_batch, uint32_t *packed_target_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, int32_t *score, int32_t *query_batch_end, int32_t *target_batch_end, int32_t *query_batch_start, int32_t *target_batch_start,int n_tasks, int32_t k_band_width) {
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
						#ifdef DEBUG
						if(tid==0)
						{
							printf("(%d, %d) - ",x, y);
						}
						#endif

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
			#ifdef DEBUG
			if(tid==0)
			printf("\n");
			#endif
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
						#ifdef DEBUG
						if(tid==0)
						{
							printf("(%d, %d) - ",x, y);
						}
						#endif

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

__global__ void gasal_banded_tiled_kernel(uint32_t *packed_query_batch, uint32_t *packed_target_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, int32_t *score, int32_t *query_batch_end, int32_t *target_batch_end, int n_tasks, int32_t k_band_width) {
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
	int32_t k_band_width_loc = k_band_width>>3;
 	int32_t k_other_band_width = (target_batch_regs - (query_batch_regs - k_band_width_loc));
	#ifdef DEBUG
	if(tid==0)
	{
		printf("k, k_other = %d , %d \n",k_band_width_loc, k_other_band_width);
	}
	#endif
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

		for (j = MAX(0, i - k_other_band_width+1)  ; j < MIN( k_band_width_loc + i, (int32_t)query_batch_regs); j++) { //query_batch sequence in columns


			#ifdef DEBUG
			if(tid==1)
			{
				if (j == MAX(0, i - k_other_band_width+1) || j == MIN( k_band_width_loc + i, (int32_t)query_batch_regs) - 1)
				printf("i,j = (%d, %d) - ",i, j);
				if ( j == MIN( k_band_width_loc + i, (int32_t)query_batch_regs) - 1)
					printf("\n");
			}
			#endif
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

	score[tid] = maxHH;//copy the max score to the output array in the GPU mem
	query_batch_end[tid] = maxXY_x;//copy the end position on query_batch sequence to the output array in the GPU mem
	target_batch_end[tid] = maxXY_y;//copy the end position on target_batch sequence to the output array in the GPU mem

	return;

}


