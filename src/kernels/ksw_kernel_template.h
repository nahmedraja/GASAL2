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




#define TILE_SIDE (8)

/* typename meaning : 
    - B is for computing the Second Best Score. Its values are on enum FALSE(0)/TRUE(1).
    (sidenote: it's based on an enum instead of a bool in order to generalize its type from its Int value, with Int2Type meta-programming-template)
*/
template <typename B>
__global__ void gasal_ksw_kernel(uint32_t *packed_query_batch, uint32_t *packed_target_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, uint32_t *seed_score, gasal_res_t *device_res, gasal_res_t *device_res_second, int n_tasks)
{
    

    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;//thread ID
	if (tid >= n_tasks) return;
	int32_t i, m;
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

	short2 HD;
	short2 initHD = make_short2(seed_score[tid], 0); //copies score from seed.
	
	uint32_t packed_target_batch_idx = target_batch_offsets[tid] >> 3; //starting index of the target_batch sequence
	uint32_t packed_query_batch_idx = query_batch_offsets[tid] >> 3;//starting index of the query_batch sequence
	uint32_t read_len = query_batch_lens[tid];
	uint32_t ref_len = target_batch_lens[tid];
	uint32_t query_batch_regs = (read_len >> 3) + (read_len&7 ? 1 : 0);//number of 32-bit words holding query_batch sequence
	uint32_t target_batch_regs = (ref_len >> 3) + (ref_len&7 ? 1 : 0);//number of 32-bit words holding target_batch sequence
    uint32_t read_len_padded = query_batch_regs << 3;
    //uint32_t ref_len_padded = target_batch_regs << 3; //unused
	//-----arrays for saving intermediate values------
	short2 global[MAX_SEQ_LEN];
	int32_t h[9];
	int32_t f[9];
	int32_t p[9];
	//--------------------------------------------
    
    // copies initialization from ksw "fill the first row", line-by-line
    global[0] = initHD;
    global[1] = make_short2(max(initHD.x - _cudaGapOE, 0) , 0);
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
    int32_t u = 0;
    register uint32_t gpac;
    register uint32_t rpac;
    int target_tile_id, query_tile_id, query_base_id, target_base_id; 
    int query_begin = 0;
    int query_end = read_len_padded; // these are the values to fix the beginning and end of the column of calculation
    int query_tile_bound, query_base_bound;

    h[0] = seed_score[tid];
    p[0] = seed_score[tid];
	for (target_tile_id = 0; target_tile_id < target_batch_regs; target_tile_id++) //target_batch sequence in rows
    {
        for (m = 1; m < 9; m++, u++) {
            h[m] = max(seed_score[tid] -(_cudaGapO + (_cudaGapExtend*(u))), 0);
            f[m] = 0;
            p[m] = max(seed_score[tid] -(_cudaGapO + (_cudaGapExtend*(u))), 0);
        }

        for (target_base_id = 0; target_base_id < TILE_SIDE; target_base_id++)
        {
			gpac = packed_target_batch[packed_target_batch_idx + target_tile_id];//load 8 packed bases from target_batch sequence
			uint32_t gbase = (gpac >> (32 - (target_base_id+1)*4 )) & 0x0F; /* get a base from target_batch sequence */ 

			query_tile_id = (query_begin / TILE_SIDE);
			query_tile_bound = (query_end / TILE_SIDE) - (query_end % TILE_SIDE == 0? 1 : 0);

            for (/*query_tile_id initialized*/; query_tile_id < query_tile_bound; query_tile_id++) //query_batch sequence in columns
            {

				if (query_tile_id == (query_begin / TILE_SIDE))
					query_base_id = query_begin % TILE_SIDE;
				else
					query_base_id = 0;

				if (query_tile_id == query_tile_bound - 1)
					query_base_bound = (query_end % TILE_SIDE) + ((query_end % TILE_SIDE) == 0)*TILE_SIDE;
				else
					query_base_bound = TILE_SIDE;

				rpac = packed_query_batch[packed_query_batch_idx + query_tile_id];//load 8 bases from query_batch sequence

                for (/*query_base_id initialized*/; query_base_id < query_base_bound; query_base_id++)
                {
					
					uint32_t rbase = (rpac >> (32 - (query_base_id+1)*4 )) & 0x0F;//get a base from query_batch sequence
                    
                    /*
                        h[target_base_id+1] = max(seed_score[tid] -(_cudaGapO + (_cudaGapExtend*(target_tile_id * TILE_SIDE + target_base_id))), 0); 
                        f[target_base_id+1] = 0;
                        p[target_base_id+1] = max(seed_score[tid] -(_cudaGapO + (_cudaGapExtend*(target_tile_id * TILE_SIDE + target_base_id))), 0); 
                    */

                    //-----load intermediate values--------------
                    HD = global[query_tile_id * TILE_SIDE + query_base_id];
                    h[target_base_id] = HD.x;
                    e = HD.y;

                    DEV_GET_SUB_SCORE_LOCAL(subScore, rbase, gbase);/* check equality of rbase and gbase */ \
                    f[target_base_id+1] = max(h[target_base_id+1]- _cudaGapOE, f[target_base_id+1] - _cudaGapExtend);/* whether to introduce or extend a gap in query_batch sequence */ \
                    h[target_base_id+1] = p[target_base_id+1] + subScore; /*score if rbase is aligned to gbase*/ \
                    h[target_base_id+1] = max(h[target_base_id+1], f[target_base_id+1]); \
                    h[target_base_id+1] = max(h[target_base_id+1], 0); \
                    e = max(h[target_base_id] - _cudaGapOE, e - _cudaGapExtend);/*whether to introduce or extend a gap in target_batch sequence */\
                    h[target_base_id+1] = max(h[target_base_id+1], e); \
                    maxXY_y = (maxHH < h[target_base_id+1]) ? (target_tile_id * TILE_SIDE + target_base_id) : maxXY_y; \
                    maxHH = (maxHH < h[target_base_id+1]) ? h[target_base_id+1] : maxHH; \
                    p[target_base_id+1] = h[target_base_id];

                    if (SAMETYPE(B, Int2Type<TRUE>))
                    {
                        bool override_second = (maxHH_second < h[target_base_id+1]) && (maxHH > h[target_base_id+1]);
                        maxXY_y_second = (override_second) ? target_tile_id * TILE_SIDE + target_base_id : maxXY_y_second; 
                        maxHH_second = (override_second) ? h[target_base_id+1] : maxHH_second;
                    }

                    //----------save intermediate values------------
                    HD.x = h[target_base_id+1];
                    HD.y = e;
                    global[query_tile_id * TILE_SIDE + query_base_id] = HD;
                    //---------------------------------------------
                
                    maxXY_x = (prev_maxHH < maxHH) ? (query_tile_id * TILE_SIDE + query_base_id) : maxXY_x;//end position on query_batch sequence corresponding to current maximum score
                    if (SAMETYPE(B, Int2Type<TRUE>))
                    {
                        maxXY_x_second = (prev_maxHH_second < maxHH) ? (query_tile_id * TILE_SIDE + query_base_id) : maxXY_x_second;
                        prev_maxHH_second = max(maxHH_second, prev_maxHH_second);
                    }
                    prev_maxHH = max(maxHH, prev_maxHH);
                } // end fo( compute query tile)
            } // end for (compute query line)
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
             
            global[query_end].x = h[target_base_id+1]; //eh[end].h = h1;
            global[query_end].y = 0; //eh[end].e = 0;
            int h_val;
            int e_val;
            HD = global[query_begin];
            h_val = HD.x;
            e_val = HD.y;
            while (query_begin < query_end && h_val == 0 && e_val == 0)
            {
                query_begin++;
                HD = global[query_begin];
                h_val = HD.x;
                e_val = HD.y;
            }
            HD = global[query_end];
            h_val = HD.x;
            e_val = HD.y;
            while (query_end >= query_begin && h_val == 0 && e_val == 0)
            {
                query_end--;
                HD = global[query_end];
                h_val = HD.x;
                e_val = HD.y;
            }
            if (query_end + 2 < ref_len)
                query_end = query_end + 2;
            else
                query_end = read_len_padded;
            

            // disable bound check for testing
            //query_begin = 0;
            query_end = read_len_padded;

        } // end for (pack of 8 bases for query)
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

    return;


}
#endif
