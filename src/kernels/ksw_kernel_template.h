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


#define PEN_CLIP5 (5)
#define TILE_SIDE (8)

/* typename meaning : 
    - B is for computing the Second Best Score. Its values are on enum FALSE(0)/TRUE(1).
    (sidenote: it's based on an enum instead of a bool in order to generalize its type from its Int value, with Int2Type meta-programming-template)
*/
/* 
    //! Note from the bwa-gasal2 coder : I failed to understand it, so I copied it.
    //! You can say to me...
    You cheated not only the game, but yourself.

    You didn't grow.
    You didn't improve.
    You took a shortcut and gained nothing.

    You experienced a hollow victory.
    Nothing was risked and nothing was gained.

    It's sad that you don't know the difference.
*/

typedef struct {
   int32_t h, e;
} eh_t;

template <typename B>
__global__ void gasal_ksw_kernel(uint32_t *packed_query_batch, uint32_t *packed_target_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, uint32_t *seed_score, gasal_res_t *device_res, gasal_res_t *device_res_second, int n_tasks)
{
    const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;//thread ID
    if (tid >= n_tasks) return;

	uint32_t packed_target_batch_idx = target_batch_offsets[tid] >> 3; //starting index of the target_batch sequence
	uint32_t packed_query_batch_idx = query_batch_offsets[tid] >> 3;//starting index of the query_batch sequence
	uint32_t qlen = query_batch_lens[tid];
	uint32_t tlen = target_batch_lens[tid];
	uint32_t query_batch_regs = (qlen >> 3) + 1;//(qlen >> 3) + (qlen & 0b0111 ? 1 : 0);//number of 32-bit words holding query_batch sequence
	uint32_t target_batch_regs = (tlen >> 3) + 1;//(tlen >> 3) + (tlen & 0b0111 ? 1 : 0);//number of 32-bit words holding target_batch sequence
    uint32_t h0 = seed_score[tid];
    int32_t subScore;
    uint32_t target_tile_id, target_base_id, query_tile_id, query_base_id, query_begin, query_tile_bound, query_base_bound;
    uint32_t gpac, rpac, gbase, rbase;
    int zdrop = 0;

    int o_del = _cudaGapO;
    int o_ins = _cudaGapO;
    int e_del = _cudaGapExtend;
    int e_ins = _cudaGapExtend;

    eh_t eh[MAX_SEQ_LEN] ; // score array
    int i, j, k, oe_del = o_del + e_del, oe_ins = o_ins + e_ins, beg, end, max, max_i, max_j, max_ie, gscore, max_off;
    for (i = 0; i < MAX_SEQ_LEN; i++)
    {
        eh[i].h = 0;
        eh[i].e = 0;
    }

    // fill the first row
    eh[0].h = h0;
    eh[1].h = h0 > oe_ins ? h0 - oe_ins : 0;
    for (j = 2; j <= qlen && eh[j - 1].h > e_ins; ++j)
        eh[j].h = eh[j - 1].h - e_ins;
   
    // DP loop
    max = h0, max_i = max_j = -1;
    max_ie = -1, gscore = -1;
    max_off = 0;
    beg = 0, end = qlen;
    
    for (target_tile_id = 0; target_tile_id < target_batch_regs; target_tile_id++) //target_batch sequence in rows
    {
        gpac = packed_target_batch[packed_target_batch_idx + target_tile_id];//load 8 packed bases from target_batch sequence

        for (target_base_id = 0; target_base_id < TILE_SIDE; target_base_id++)
        {
        /*
        for (i = 0; (i < tlen); ++i) 
        {
            target_tile_id = i / TILE_SIDE;
            target_base_id = i % TILE_SIDE;
        */
            i = target_tile_id * TILE_SIDE + target_base_id;

            if (i >= tlen) // skip padding
                break;
            
            //gpac = packed_target_batch[packed_target_batch_idx + target_tile_id];//load 8 packed bases from target_batch sequence
            gbase = (gpac >> (32 - (target_base_id+1)*4 )) & 0x0F; /* get a base from target_batch sequence */

            int t, f = 0, h1, m = 0, mj = -1;
            // compute the first column
            if (beg == 0) {
                h1 = h0 - (o_del + e_del * (i + 1));
                if (h1 < 0)
                h1 = 0;
            } else
                h1 = 0;
            
            //for (j = beg; (j < end); ++j) {
            // FIXME: could be a problem with borderline cases like 1 to 7 bases only (explaining the very small difference)

            
            for(query_tile_id = 0; (query_tile_id < query_batch_regs); query_tile_id++)
            {
                rpac = packed_query_batch[packed_query_batch_idx + query_tile_id];//load 8 bases from query_batch sequence

                for(query_base_id = 0; (query_base_id < TILE_SIDE); query_base_id++)
                {
                    j = query_tile_id * TILE_SIDE + query_base_id;
                    //query_tile_id = j / TILE_SIDE;
                    //query_base_id = j % TILE_SIDE;
                    if (j < beg)
                        continue;      
                    if (j >= end)
                        break;


                    rbase = (rpac >> (32 - (query_base_id+1)*4 )) & 0x0F;//get a base from query_batch sequence

                    // At the beginning of the loop: eh[j] = { H(i-1,j-1), E(i,j) }, f = F(i,j) and h1 = H(i,j-1)
                    // Similar to SSE2-SW, cells are computed in the following order:
                    //   H(i,j)   = max{H(i-1,j-1)+S(i,j), E(i,j), F(i,j)}
                    //   E(i+1,j) = max{H(i,j)-gapo, E(i,j)} - gape
                    //   F(i,j+1) = max{H(i,j)-gapo, F(i,j)} - gape
                    eh_t *p = &eh[j];
                    int h, M = p->h, e = p->e; // get H(i-1,j-1) and E(i-1,j)
                    p->h = h1;          // set H(i,j-1) for the next row
                    subScore = (rbase == gbase) ? _cudaMatchScore : -_cudaMismatchScore;
                    subScore = ((rbase == N_VALUE) || (gbase == N_VALUE)) ? -N_PENALTY : subScore;
                    M = M ? M + subScore : 0;          // separating H and M to disallow a cigar like "100M3I3D20M"
                    h = M > e ? M : e;   // e and f are guaranteed to be non-negative, so h>=0 even if M<0
                    h = h > f ? h : f;
                    h1 = h;             // save H(i,j) to h1 for the next column
                    mj = m > h ? mj : j; // record the position where max score is achieved
                    m = m > h ? m : h;   // m is stored at eh[mj+1]
                    t = M - oe_del;
                    t = t > 0 ? t : 0;
                    e -= e_del;
                    e = e > t ? e : t;   // computed E(i+1,j)
                    p->e = e;           // save E(i+1,j) for the next row
                    t = M - oe_ins;
                    t = t > 0 ? t : 0;
                    f -= e_ins;
                    f = f > t ? f : t;   // computed F(i,j+1)
                }
            }
            eh[end].h = h1;
            eh[end].e = 0;
            if (j == qlen) {
                max_ie = gscore > h1 ? max_ie : i;
                gscore = gscore > h1 ? gscore : h1;
            }
            if (m == 0)
                break;
            if (m > max) {
                max = m, max_i = i, max_j = mj;
                max_off = max_off > abs(mj - i) ? max_off : abs(mj - i);
            } else if (zdrop > 0) {
                if (i - max_i > mj - max_j) {
                if (max - m - ((i - max_i) - (mj - max_j)) * e_del > zdrop)
                    break;
                } else {
                if (max - m - ((mj - max_j) - (i - max_i)) * e_ins > zdrop)
                    break;
                }
            }
            /* This is defining from where to start the next row and where to end the computation of next row
                it skips some of the cells in the beginning and in the end of the row
            */
            // update beg and end for the next round
            // COULD be done over a constant value...
            for (j = beg; (j < end) && eh[j].h == 0 && eh[j].e == 0; ++j)
                ;
            beg = j;
            for (j = end; (j >= beg) && eh[j].h == 0 && eh[j].e == 0; --j)
                ;
            end = j + 2 < qlen ? j + 2 : qlen;
            //beg = 0; end = qlen; // uncomment this line for debugging
        }
    }

    if (gscore <= 0 || gscore <= max - PEN_CLIP5)
    {
        device_res->aln_score[tid] = max;
        device_res->query_batch_end[tid] = max_j + 1;
        device_res->target_batch_end[tid] = max_i + 1;
    } else {
        device_res->aln_score[tid] = gscore;
        device_res->query_batch_end[tid] = qlen;
        device_res->target_batch_end[tid] = max_ie + 1;
    }

}


#endif

