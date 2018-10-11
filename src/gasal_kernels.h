//#define DEBUG

#ifdef DEBUG
#include <stdio.h>
#endif


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



#include "gasal_kernels_seqpak.h"
#include "gasal_kernels_global.h"
#include "gasal_kernels_semiglobal.h"
#include "gasal_kernels_local.h"
#include "gasal_kernels_banded.h"
