#ifndef __KSW_KERNEL_TEMPLATE__
#define __KSW_KERNEL_TEMPLATE__


// This old core provides the same result as the currently LOCAL core, but lacks some optimization. Left for historical / comparative purposes.
#define CORE_LOCAL_DEPRECATED_COMPUTE() \
    uint32_t gbase = (gpac >> l) & 15;/*get a base from target_batch sequence */ \
    DEV_GET_SUB_SCORE_LOCAL(subScore, rbase, gbase);/* check equality of rbase and gbase */ \
    f[m] = max(h[m]- _cudaGapOE, f[m] - _cudaGapExtend);/* whether to introduce or extend a gap in query_batch sequence */ \
    h[m] = p[m] + subScore; /*score if rbase is aligned to gbase*/ \
    h[m] = max(h[m], f[m]); \
    h[m] = max(h[m], 0); \
    e = max(h[m - 1] - _cudaGapOE, e - _cudaGapExtend);/*whether to introduce or extend a gap in target_batch sequence */\
    h[m] = max(h[m], e); \
    maxXY_y = (maxHH < h[m]) ? gidx + (m-1) : maxXY_y; \
    maxHH = (maxHH < h[m]) ? h[m] : maxHH; \
    p[m] = h[m-1];





/* typename meaning : 
    - B is for computing the Second Best Score. Its values are on enum FALSE(0)/TRUE(1).
    (sidenote: it's based on an enum instead of a bool in order to generalize its type from its Int value, with Int2Type meta-programming-template)
*/

// Define a fake base score, to avoid passing the seed length
#define BASE_SCORE_SEED (30)

template <typename B>
__global__ void gasal_ksw_kernel(uint32_t *packed_query_batch, uint32_t *packed_target_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, gasal_res_t *device_res, gasal_res_t *device_res_second, int n_tasks)
{
    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;//thread ID
	if (tid >= n_tasks) return;

	int32_t i, j, k, m, l;
	int32_t e;

    int32_t maxHH = 0; //initialize the maximum score to zero
	int32_t maxXY_y = 0; 

    int32_t prev_maxHH = 0;
    int32_t maxXY_x = 0;    


    int32_t maxHH_second __attribute__((unused)); // __attribute__((unused)) to avoid raising errors at compilation. most template-kernels don't use these.
    int32_t prev_maxHH_second __attribute__((unused)); 
    int32_t maxXY_x_second __attribute__((unused));
    int32_t maxXY_y_second __attribute__((unused));
    maxHH_second = 0;
    prev_maxHH_second = 0;
    maxXY_x_second = 0;
    maxXY_y_second = 0;


	int32_t subScore;

	int32_t ridx, gidx;
	short2 HD;
	short2 initHD = make_short2(BASE_SCORE_SEED, 0);
	
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
    
    // copies initialization from ksw "fill the first row", line-by-line
    global[0] = initHD;
    global[1] = make_short2((initHD.x > _cudaGapOE) ? (initHD.x - _cudaGapOE) : (0) , 0);
    for (i = 2; i < MAX_SEQ_LEN; i++) {
        initHD = make_short2(max(initHD.x - _cudaGapExtend, 0) , 0);
        global[i] = initHD;
    }
    /*
        // begin / end to skip some stuff. BUT TILES FUCK
        // So let's forget about it.
        int beg = 0, end = ref_len;
    */
    // bwa does: for (j = beg; LIKELY(j < end); ++j)
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


                register int32_t prev_hm_diff = h[0] - _cudaGapOE;
                #pragma unroll 8
                for (l = 28, m = 1; m < 9; l -= 4, m++) {
                            uint32_t gbase = (gpac >> l) & 15; /* get a base from target_batch sequence */ 
                            DEV_GET_SUB_SCORE_LOCAL(subScore, rbase, gbase);/* check equality of rbase and gbase */
                            register int32_t curr_hm_diff = h[m] - _cudaGapOE;
                            f[m] = max(curr_hm_diff, f[m] - _cudaGapExtend);/* whether to introduce or extend a gap in query_batch sequence */
                            curr_hm_diff = p[m] + subScore;/* score if rbase is aligned to gbase */
                            curr_hm_diff = max(curr_hm_diff, f[m]);
                            curr_hm_diff = max(curr_hm_diff, 0);
                            e = max(prev_hm_diff, e - _cudaGapExtend);/* whether to introduce or extend a gap in target_batch sequence */
                            curr_hm_diff = max(curr_hm_diff, e);
                            maxXY_y = (maxHH < curr_hm_diff) ? gidx + (m-1) : maxXY_y; 
                            maxHH = (maxHH < curr_hm_diff) ? curr_hm_diff : maxHH;
                            h[m] = curr_hm_diff;
                            p[m] = prev_hm_diff + _cudaGapOE;
                            prev_hm_diff=curr_hm_diff - _cudaGapOE;
                        if (SAMETYPE(B, Int2Type<TRUE>))
                        {
                            bool override_second = (maxHH_second < curr_hm_diff) && (maxHH > curr_hm_diff);
                            maxXY_y_second = (override_second) ? gidx + (m-1) : maxXY_y_second; 
                            maxHH_second = (override_second) ? curr_hm_diff : maxHH_second;
                        }
                }

                //----------save intermediate values------------
                HD.x = h[m-1];
                HD.y = e;
                global[ridx] = HD;
                //---------------------------------------------
            

                maxXY_x = (prev_maxHH < maxHH) ? ridx : maxXY_x;//end position on query_batch sequence corresponding to current maximum score

                if (SAMETYPE(B, Int2Type<TRUE>))
                {
                    maxXY_x_second = (prev_maxHH_second < maxHH) ? ridx : maxXY_x_second;
                    prev_maxHH_second = max(maxHH_second, prev_maxHH_second);
                }
                prev_maxHH = max(maxHH, prev_maxHH);
                ridx++;
                //-------------------------------------------------------

            } // end for (compute tile)
        } // end for (pack of 8 bases for query)

        /* This is defining from where to start the next row and where to end the computation of next row
         it skips some of the cells in the beginning and in the end of the row
        */
       /*
        for (j = beg; j < end && eh[j].h == 0 && eh[j].e == 0; ++j)
            ;
        beg = j;
        for (j = end; j >= beg && eh[j].h == 0 && eh[j].e == 0; --j)
            ;
        end = j + 2 < ref_len ? j + 2 : query_len;
        */

	} // end for (pack of 8 bases for target)

	device_res->aln_score[tid] = maxHH;//copy the max score to the output array in the GPU mem
	device_res->query_batch_end[tid] = maxXY_x;//copy the end position on query_batch sequence to the output array in the GPU mem
	device_res->target_batch_end[tid] = maxXY_y;//copy the end position on target_batch sequence to the output array in the GPU mem

    if (SAMETYPE(B, Int2Type<TRUE>))
    {
        device_res_second->aln_score[tid] = maxHH_second;
        device_res_second->query_batch_end[tid] = maxXY_x_second;
        device_res_second->target_batch_end[tid] = maxXY_y_second;
    }


    // 
    return;


}
#endif
