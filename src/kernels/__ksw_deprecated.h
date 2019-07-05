
template <typename B>
__global__ void gasal_ksw2_kernel(uint32_t *packed_query_batch, uint32_t *packed_target_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, uint32_t *seed_score, gasal_res_t *device_res, gasal_res_t *device_res_second, int n_tasks)
{
    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;//thread ID
	if (tid >= n_tasks) return;
	int32_t i, j, m;
	int32_t e;

    int32_t maxHH = -1; //initialize the maximum score to zero
	int32_t maxXY_y = -1; 

    int32_t prev_maxHH = -1;
    int32_t maxXY_x = -1;    


    int32_t maxHH_second __attribute__((unused)); // __attribute__((unused)) to avoid raising errors at compilation. most template-kernels don't use these.
    int32_t prev_maxHH_second __attribute__((unused)); 
    int32_t maxXY_x_second __attribute__((unused));
    int32_t maxXY_y_second __attribute__((unused));
    maxHH_second = -1;
    prev_maxHH_second = -1;
    maxXY_x_second = -1;
    maxXY_y_second = -1;

	int32_t subScore = 0;

	short2 HD;
	short2 initHD = make_short2(seed_score[tid], 0); //copies score from seed.

	uint32_t packed_target_batch_idx = target_batch_offsets[tid] >> 3; //starting index of the target_batch sequence
	uint32_t packed_query_batch_idx = query_batch_offsets[tid] >> 3;//starting index of the query_batch sequence
	uint32_t query_length = query_batch_lens[tid];
	uint32_t target_length = target_batch_lens[tid];
	uint32_t query_batch_regs = (query_length >> 3) + (query_length&7 ? 1 : 0);//number of 32-bit words holding query_batch sequence
	uint32_t target_batch_regs = (target_length >> 3) + (target_length&7 ? 1 : 0);//number of 32-bit words holding target_batch sequence
	//-----arrays for saving intermediate values------
	short2 global[MAX_QUERY_LEN];

	//--------------------------------------------
    // copies initialization from ksw "fill the first row", line-by-line
    
    global[0] = make_short2(seed_score[tid] , 0);
    global[1] = make_short2(max(seed_score[tid] - _cudaGapOE, 0) , 0);


    int32_t h0 = seed_score[tid];


    int32_t u = 0;
    register uint32_t gpac;
    register uint32_t rpac;
    int target_tile_id, query_tile_id, query_base_id, target_base_id; 
    int query_begin = 0;
    int query_end = query_length; // these are the values to fix the beginning and end of the column of calculation
    int global_score = -1;
    int global_target_end = -1;

    int max, max_ie, max_i, max_j, gscore;

	int32_t h[9];
	int32_t f[9];
	int32_t p[9];
    h[0] = seed_score[tid];
    p[0] = seed_score[tid];
    int query_tile_bound, query_base_bound, query_id, target_id;

    for (i = 2; i < MAX_QUERY_LEN; i++)
    {
        global[i].x = 0;
        global[i].y = 0;
    }
    // fill the first row
    global[0].x = h0;
    global[1].x = h0 > _cudaGapOE ? h0 - _cudaGapOE : 0;
    for (j = 2; j <= query_length && global[j - 1].x > _cudaGapExtend; ++j)
    {
        global[j].x = global[j-1].x - _cudaGapExtend;
    }
    
	for (target_tile_id = 0; target_tile_id < target_batch_regs; target_tile_id++) //target_batch sequence in rows
    {

        u--;
        for (m = 0; m < 9; m++, u++) {
            h[m] = max(seed_score[tid] -(_cudaGapO + (_cudaGapExtend*(u))), 0);
            f[m] = 0;
            p[m] = max(seed_score[tid] -(_cudaGapO + (_cudaGapExtend*(u))), 0);
        }
        gpac = packed_target_batch[packed_target_batch_idx + target_tile_id];//load 8 packed bases from target_batch sequence

        for (target_base_id = 0; target_base_id < TILE_SIDE; target_base_id++)
        {
			query_tile_id = (query_begin / TILE_SIDE);
			query_tile_bound = (query_end / TILE_SIDE);
            uint32_t gbase = (gpac >> (32 - (target_base_id+1)*4 )) & 0x0F; 

            for (; query_tile_id < query_tile_bound; query_tile_id++)
            {
				if (query_tile_id == (query_begin / TILE_SIDE))
					query_base_id = query_begin % TILE_SIDE;
				else
					query_base_id = 0;

				if (query_tile_id == query_tile_bound - 1)
					query_base_bound = (query_end % TILE_SIDE);
				else
					query_base_bound = TILE_SIDE;

                rpac = packed_query_batch[packed_query_batch_idx + query_tile_id];//load 8 bases from query_batch sequence
                for ( ; query_base_id < query_base_bound; query_base_id++)
                {                    
                    uint32_t rbase = (rpac >> (32 - (query_base_id+1)*4 )) & 0x0F;//get a base from query_batch sequence

                    //-----load intermediate values--------------
                    HD = global[query_tile_id * TILE_SIDE + query_base_id];
                    h[target_base_id] = HD.x;
                    e = HD.y;

                    subScore = (rbase == gbase) ?_cudaMatchScore : -_cudaMismatchScore;
	                subScore = ((rbase == N_VALUE) || (gbase == N_VALUE)) ? -N_PENALTY : subScore;
                    f[target_base_id+1] = max(h[target_base_id+1]- _cudaGapOE, f[target_base_id+1] - _cudaGapExtend);
                    h[target_base_id+1] = p[target_base_id+1] + subScore; 
                    h[target_base_id+1] = max(h[target_base_id+1], f[target_base_id+1]); 
                    h[target_base_id+1] = max(h[target_base_id+1], 0); 
                    e = max(h[target_base_id] - _cudaGapOE, e - _cudaGapExtend);
                    h[target_base_id+1] = max(h[target_base_id+1], e); 
                    maxXY_y = (maxHH < h[target_base_id+1]) ? (target_tile_id * TILE_SIDE + target_base_id) : maxXY_y; 
                    maxHH = (maxHH < h[target_base_id+1]) ? h[target_base_id+1] : maxHH; 
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
            } // end for (compute all query tiles, so the whole query line)

            // store global_score
            if (query_tile_id * TILE_SIDE + query_base_id >= query_length) 
            {
                global_target_end = global_score > h[target_base_id+1] ? global_target_end : target_tile_id*TILE_SIDE + target_base_id;
                global_score = max(global_score, h[target_base_id+1]);
            }


            // This is defining from where to start the next row and where to end the computation of next row
            //    it skips some of the cells in the beginning and in the end of the row
            

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
            if (query_end + 2 < query_length)
                query_end = query_end + 2;
            else
                query_end = query_length;
            
            // disable bound check for testing
            //query_begin = 0;
            // query_end = query_length_padded;
            
        } // end for (pack of 8 bases for query)
	} // end for (pack of 8 bases for target)

    if (global_score <= 0 || global_score <= maxHH - PEN_CLIP5)
    {
        device_res->aln_score[tid] = maxHH;
        device_res->query_batch_end[tid] = maxXY_x + 1;
        device_res->target_batch_end[tid] = maxXY_y + 1;
    } else {
        device_res->aln_score[tid] = global_score;
        device_res->query_batch_end[tid] = query_length;
        device_res->target_batch_end[tid] = global_target_end + 1;
    }
    return;
}

/*
template <typename B>
__global__ void gasal_ksw_kernel(uint32_t *packed_query_batch, uint32_t *packed_target_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, uint32_t *seed_score, gasal_res_t *device_res, gasal_res_t *device_res_second, int n_tasks)
{
    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;//thread ID
	if (tid >= n_tasks) return;
	int32_t i, j;
	int32_t e;

    int32_t maxHH = -1; //initialize the maximum score to zero
	int32_t maxXY_y = -1; 

    int32_t prev_maxHH = -1;
    int32_t maxXY_x = -1;    


    int32_t maxHH_second __attribute__((unused)); // __attribute__((unused)) to avoid raising errors at compilation. most template-kernels don't use these.
    int32_t prev_maxHH_second __attribute__((unused)); 
    int32_t maxXY_x_second __attribute__((unused));
    int32_t maxXY_y_second __attribute__((unused));
    maxHH_second = -1;
    prev_maxHH_second = -1;
    maxXY_x_second = -1;
    maxXY_y_second = -1;

	int32_t subScore = 0;

	short2 HD;
	short2 initHD = make_short2(seed_score[tid], 0); //copies score from seed.

	uint32_t packed_target_batch_idx = target_batch_offsets[tid] >> 3; //starting index of the target_batch sequence
	uint32_t packed_query_batch_idx = query_batch_offsets[tid] >> 3;//starting index of the query_batch sequence
	uint32_t query_length = query_batch_lens[tid];
	uint32_t target_length = target_batch_lens[tid];
	uint32_t query_batch_regs = (query_length >> 3) + (query_length&7 ? 1 : 0);//number of 32-bit words holding query_batch sequence
	uint32_t target_batch_regs = (target_length >> 3) + (target_length&7 ? 1 : 0);//number of 32-bit words holding target_batch sequence
    //uint32_t query_length_padded = query_batch_regs << 3; //unused
    //uint32_t target_length_padded = target_batch_regs << 3; //unused
	//-----arrays for saving intermediate values------
	short2 global[MAX_QUERY_LEN];

	//--------------------------------------------
    // copies initialization from ksw "fill the first row", line-by-line
    
    global[0] = make_short2(seed_score[tid] , 0);
    global[1] = make_short2(max(seed_score[tid] - _cudaGapOE, 0) , 0);
    for (i = 2; i < MAX_QUERY_LEN; i++)
    {
        global[i].x = max(global[i].x - _cudaGapExtend, 0);
        global[i].y = 0;
    }

    int32_t h0 = seed_score[tid];
    // fill the first row
    
    global[0].x = h0;
    global[1].x = h0 > _cudaGapOE ? h0 - _cudaGapOE : 0;
    for (j = 2; j <= query_length && global[j - 1].x > _cudaGapExtend; ++j)
    {
        global[j].x = global[j-1].x - _cudaGapExtend;
        global[j].y = 0;
    }

    int32_t u = 0;
    register uint32_t gpac;
    register uint32_t rpac;
    int target_tile_id, query_tile_id, query_base_id, target_base_id; 
    int query_begin = 0;
    int query_end = query_length; // these are the values to fix the beginning and end of the column of calculation
    int global_score = -1;
    int global_target_end = -1;

    int max, max_ie, max_i, max_j, gscore;
    max = h0;
    max_i = -1;
    max_j = -1;
    max_ie = -1;
    gscore = -1;
    for (target_id = 0; target_id < target_length; ++target_id)
    {
        int t;
        int f = 0; 
        int h1;
        int m = 0;
        int mj = -1;
        
        // compute the first column
        if (query_begin == 0) {
            h1 = h0 - (_cudaGapO + _cudaGapExtend * (target_id + 1));
            if (h1 < 0)
                h1 = 0;
        }

        target_tile_id = target_id / TILE_SIDE;
        target_base_id = target_id % TILE_SIDE;
        gpac = packed_target_batch[packed_target_batch_idx + target_tile_id];
        uint32_t target_base = (gpac >> (32 - (target_base_id+1)*4 )) & 0x0F; // get a base from target_batch sequence  

        for(query_id = query_begin; query_id < query_end; ++query_id)
        {
            query_tile_id = query_id / TILE_SIDE;
            query_base_id = query_id % TILE_SIDE;
            rpac = packed_query_batch[packed_query_batch_idx + query_tile_id];//load 8 bases from query_batch sequence
            uint32_t query_base = (rpac >> (32 - (query_base_id+1)*4 )) & 0x0F;//get a base from query_batch sequence

            int h;      
            int M = global[query_id].x;
            e = global[query_id].y;  // get H(i-1,j-1) and E(i-1,j)
            global[query_id].x = h1;  // set H(i,j-1) for the next row
            subScore = (query_base == target_base) ? _cudaMatchScore : -_cudaMismatchScore;
	        subScore = ((query_base == N_VALUE) || (target_base == N_VALUE)) ? 0 : subScore;

            M = M ? M + subScore : 0;  // separating H and M to disallow a cigar like "100M3I3D20M"

            h = M > e ? M : e;   // e and f are guaranteed to be non-negative, so h>=0 even if M<0
            h = h > f ? h : f;
            h1 = h;             // save H(i,j) to h1 for the next column
            mj = m > h ? mj : query_id; // record the position where max score is achieved
            m = m > h ? m : h;   // m is stored at eh[mj+1]
            t = M - _cudaGapOE;
            t = t > 0 ? t : 0;
            e -= _cudaGapExtend;
            e = e > t ? e : t;   // computed E(i+1,j)
            global[query_id].y = e;     //p->e = e;      // save E(i+1,j) for the next row
            t = M - _cudaGapOE;
            t = t > 0 ? t : 0;
            f -= _cudaGapExtend;
            f = f > t ? f : t;   // computed F(i,j+1)
        }
        global[query_end].x = h1; //eh[end].h = h1;
        global[query_end].y = 0; //eh[end].e = 0;

        if (query_id == query_length) {
            max_ie = gscore > h1 ? max_ie : target_id;
            gscore = gscore > h1 ? gscore : h1;
        }
        if (m == 0)
        {
            //break;
        } else 
        if (m > max) {
            max = m;
            max_i = target_id;
            max_j = mj;
        }

        for (j = query_begin; (j < query_end) && ((global[j].x == 0) && (global[j].y  == 0)); ++j)
            ;
        query_begin = j;
        for (j = query_end; (j >= query_begin) && ((global[j].x == 0) && (global[j].y  == 0)); --j)
            ;
        query_end = j + 2 < query_length ? j + 2 : query_length;
    }
    //renaming


    //max = h0;
    //max_j = query_length;
    //max_i = target_length;
    // printf("tid=%d\tmax=%d\tmax_j=%d\tmax_i=%d\n", tid, max, max_j, max_i);
    //gscore = 0;
    //max_ie = 0;

    global_score = gscore;
    global_target_end = max_ie + 1;

    maxXY_x = max_j + 1;
    maxXY_y = max_i + 1;
    maxHH = max;
    


    //Penclip handling
    // check whether we prefer to reach the end of the query
    
    if (global_score <= 0 || global_score <= maxHH - PEN_CLIP5)
    {
        device_res->aln_score[tid] = maxHH;
        device_res->query_batch_end[tid] = maxXY_x;
        device_res->target_batch_end[tid] = maxXY_y;
    } else {
        device_res->aln_score[tid] = global_score;
        device_res->query_batch_end[tid] = query_length;
        device_res->target_batch_end[tid] = global_target_end;
    }

    if (SAMETYPE(B, Int2Type<TRUE>))
    {
        device_res_second->aln_score[tid] = maxHH_second;
        device_res_second->query_batch_end[tid] = maxXY_x_second;
        device_res_second->target_batch_end[tid] = maxXY_y_second;
    }

    return;
    
}
*/

#define ORIG
    #ifdef ORIG
	int32_t h[9];
	int32_t f[9];
	int32_t p[9];
    h[0] = seed_score[tid];
    p[0] = seed_score[tid];
    int query_tile_bound, query_base_bound, query_id, target_id;
    #endif


    #ifdef ORIG
	for (target_tile_id = 0; target_tile_id < target_batch_regs; target_tile_id++) //target_batch sequence in rows
    {

        if (tid==0) printf("target_tile_id=%d/%d (target_length=%d)\n", target_tile_id, target_batch_regs, target_length);
        u--;
        for (m = 0; m < 9; m++, u++) {
            h[m] = max(seed_score[tid] -(_cudaGapO + (_cudaGapExtend*(u))), 0);
            if (tid==0) printf("h[%d]=%d\t", m, h[m]);
            f[m] = 0;
            p[m] = max(seed_score[tid] -(_cudaGapO + (_cudaGapExtend*(u))), 0);
        }
        if (tid==0) printf("\n");
        gpac = packed_target_batch[packed_target_batch_idx + target_tile_id];//load 8 packed bases from target_batch sequence

        for (target_base_id = 0; target_base_id < TILE_SIDE; target_base_id++)
        {
			

            if (tid==0) printf("query_begin=%d, query_end=%d\n", query_begin, query_end);

			query_tile_id = (query_begin / TILE_SIDE);
			query_tile_bound = (query_end / TILE_SIDE);
            uint32_t gbase = (gpac >> (32 - (target_base_id+1)*4 )) & 0x0F; 

            for (; query_tile_id < query_tile_bound; query_tile_id++)
            {
				if (query_tile_id == (query_begin / TILE_SIDE))
					query_base_id = query_begin % TILE_SIDE;
				else
					query_base_id = 0;

				if (query_tile_id == query_tile_bound - 1)
					query_base_bound = (query_end % TILE_SIDE);
				else
					query_base_bound = TILE_SIDE;

                rpac = packed_query_batch[packed_query_batch_idx + query_tile_id];//load 8 bases from query_batch sequence
                for ( ; query_base_id < query_base_bound; query_base_id++)
                {                    
                    uint32_t rbase = (rpac >> (32 - (query_base_id+1)*4 )) & 0x0F;//get a base from query_batch sequence
                    


                    //-----load intermediate values--------------
                    HD = global[query_tile_id * TILE_SIDE + query_base_id];
                    h[target_base_id] = HD.x;
                    e = HD.y;

                    DEV_GET_SUB_SCORE_LOCAL(subScore, rbase, gbase); 
                    f[target_base_id+1] = max(h[target_base_id+1]- _cudaGapOE, f[target_base_id+1] - _cudaGapExtend);
                    h[target_base_id+1] = p[target_base_id+1] + subScore; 
                    h[target_base_id+1] = max(h[target_base_id+1], f[target_base_id+1]); 
                    h[target_base_id+1] = max(h[target_base_id+1], 0); 
                    e = max(h[target_base_id] - _cudaGapOE, e - _cudaGapExtend);
                    h[target_base_id+1] = max(h[target_base_id+1], e); 
                    maxXY_y = (maxHH < h[target_base_id+1]) ? (target_tile_id * TILE_SIDE + target_base_id) : maxXY_y; 
                    maxHH = (maxHH < h[target_base_id+1]) ? h[target_base_id+1] : maxHH; 
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
            } // end for (compute all query tiles, so the whole query line)

            // store global_score
            if (query_tile_id * TILE_SIDE + query_base_id >= query_length) 
            {
                global_target_end = global_score > h[target_base_id+1] ? global_target_end : target_tile_id*TILE_SIDE + target_base_id;
                global_score = max(global_score, h[target_base_id+1]);
            }


            // This is defining from where to start the next row and where to end the computation of next row
            //    it skips some of the cells in the beginning and in the end of the row
            
            
            //for (j = beg; j < end && eh[j].h == 0 && eh[j].e == 0; ++j)
            //    ;
            //beg = j;
            //for (j = end; j >= beg && eh[j].h == 0 && eh[j].e == 0; --j)
            //    ;
            //end = j + 2 < target_length ? j + 2 : query_len;
            

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
            if (query_end + 2 < query_length)
                query_end = query_end + 2;
            else
                query_end = query_length;
            
            // disable bound check for testing
            //query_begin = 0;
            // query_end = query_length_padded;
            
        } // end for (pack of 8 bases for query)
	} // end for (pack of 8 bases for target)
    #endif
    
