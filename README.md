
# GASAL2

GASAL2 is an easy to use CUDA library for DNA/RNA sequence alignment algorithms. Currently it the following sequence alignment functions.
- Local alignment without start position computation. Gives alignment score and end position of the alignment.
- Local alignment with start position computation. Gives alignment score and end and start position of the alignment.
- Semi-global alignment without start position computation. Gives score and end position of the alignment.
- Semi-global alignment with start position computation. Gives alignment score and end and start position of the alignment.
- Global alignment.

It is an extension of GASAL (https://github.com/nahmedraja/GASAL) and allows full overlapping of CPU and GPU execution

## Requirements
CUDA toolkit 8 or higher. May be 7 will also work, but not tested yet. 

## Compiling GASAL2
To compile the library, run the following two commands following commands:

```
$ ./configure.sh <path to cuda installation directory>
$ make GPU_SM_ARCH=<GPU SM architecture> MAX_LEN=<maximum sequence length> N_CODE=<code for "N", e.g. 0x4E if the bases are represented by ASCII characters> [N_PENALTY=<penalty for aligning "N" against any other base>]
```

`N_PENALTY` is optional and if it is not specified then GASAL2 considers "N" as an ordinary base having the same match/mismatch scores as for A, C, G or T. As a result of these commands, *include* and *lib* directories will be created containing `gasal.h` and `libgasal.a`, respectively. Include `gasal.h` in your code link it with `libgasal.a` during compilation. Also link the CUDA runtime library by adding `-lcudart` flag. The path to the CUDA runtime library must also be specfied while linking as *-L <path to CUDA lib64 directory>*.

## Using GASAL2
To use GASAL2  alignment functions, first the match/mismatach scores and gap open/extension penalties need to be passed on to the GPU. Assign the values match/mismatach scores and gap open/extension penalties to the members of `gasal_subst_scores` struct

```
typedef struct{
	int32_t match;
	int32_t mismatch;
	int32_t gap_open;
	int32_t gap_extend;
}gasal_subst_scores;
```

The values are passed to the GPU by calling `gasal_copy_subst_scores()` function

```
void gasal_copy_subst_scores(gasal_subst_scores *subst);
```

A vector of `gasal_gpu_storage_t` is defined using the following function:

```
gasal_gpu_storage_v gasal_init_gpu_storage_v(int n_streams);
```

`n_streams` is the number of outstanding GPU kernel launces known as *streams* . The return type is `gasal_gpu_storage_v`:

```
typedef struct{
	int n;
	gasal_gpu_storage_t *a;
}gasal_gpu_storage_v;
```

`n = n_streams` and `a` pointer to the array. An element of the array holds the required data structure of a stream. To destroy the vector the following function is used.

```
void gasal_destroy_gpu_storage_v(gasal_gpu_storage_v *gpu_storage_vec);
```

The streams in the vector are initialized by calling:

```
void gasal_init_streams(gasal_gpu_storage_v *gpu_storage_vec, int host_max_query_batch_bytes,  int gpu_max_query_batch_bytes,  int host_max_target_batch_bytes, int gpu_max_target_batch_bytes, int host_max_n_alns, int gpu_max_n_alns, int algo, int start);
```

In GASAL2, the sequences to be alignned are conatined in two batches i.e. a sequence in query_batch is aligned to sequence in target_batch. A *batch* is a concatenation of sequences. *The number of bases in each sequence must a multiple of 8*. Hence, if a sequence is not a multiple of 8, `N's` are added at the end of sequence. We call these redundant bases as *Pad bases*. Note that the pad bases are always "N's" irrespective of whether `N_PENALTY` is defined or not.. With the help of *max_batch_bytes* the user specifies the expected maxumum size(in bytes) of sequences in the two batches. *host_max_batch_bytes* are pre-allocated in the CPU memory. Smilarly, *gpu_max_batch_bytes* are pre-allocated in the GPU memory. *max_n_alns* is the expected number of sequences to be aligned. If the actual required GPU memory is more than the pre-allocated memory, GASAL2 automatically allocates more memory. This is not true for the memory allocated on the CPU side. The number of sequences and the size of batches must not exceed *host_max_n_alns* and *host_max_batch_bytes*, respectively. This requirement may be removed in the future.  The type of sequence alignment algorithm is specfied using `algo` parameter. Pass one of the follwing three values as the `algo` parameter:

```
LOCAL
GLOBAL
SEMI_GLOBAL
```

Similarly, to perform alignment with or without start position, computation is specfied by passing one the following two values in the `start` parameter:

```
WITHOUT_START
WITH_START
```

To free up the allocated memory the following function is used:

```
void gasal_destroy_streams(gasal_gpu_storage_v *gpu_storage_vec);
```

The `gasal_init_streams()` and `gasal_destroy_streams()` internally use `cudaMalloc()`, `cudaMallocHost()`, `cudaFree()` and `cudaFreeHost()` functions. These CUDA API functions are time expensive. Therefore, `gasal_init_streams()` and `gasal_destroy_streams()` should be preferably called only once in the program.

The `gasal_gpu_storage_t` holds the data structures for a stream. In the following we only show those members of `gasal_gpu_storage_t` which are accessed by the user:

```
typedef struct{
	...
	uint8_t *host_unpacked_query_batch;
	uint8_t *host_unpacked_target_batch;
	uint32_t *host_query_batch_offsets;
	uint32_t *host_target_batch_offsets;
	uint32_t *host_query_batch_lens;
	uint32_t *host_target_batch_lens;
	int32_t *host_aln_score;
	int32_t *host_query_batch_end;
	int32_t *host_target_batch_end;
	int32_t *host_query_batch_start;
	int32_t *host_target_batch_start;
	uint32_t host_max_query_batch_bytes;
	uint32_t host_max_target_batch_bytes;
	uint32_t host_max_n_alns;
	int is_free;
	...

} gasal_gpu_storage_t;
```

To align the sequences the user first need to check the availability of the stream. If `is_free` is  1 the user can use the current stream to perform the alignment on the GPU. To do this, `host_unpacked_query_batch` and `host_unpacked_target_batch` are filled with the sequences to be aligned. Th user makes sure that the number of sequences and the size of batches must not exceed *host_max_n_alns* and *host_max_batch_bytes*, respectively. `host_query_batch_offsets` and `host_target_batch_offsets` contain the starting point of sequences in the batch that are required to be aligned. These offset values include the pad bases, and hence always a multiple of 8. `host_query_batch_lens` and `host_target_batch_lens` are the original length of sequences i.e. excluding pad bases. Alignment is performed by calling the follwing function:

```
void gasal_aln_async(gasal_gpu_storage_t *gpu_storage, const uint32_t actual_query_batch_bytes, const uint32_t actual_target_batch_bytes, const uint32_t actual_n_alns, int algo, int start);
```


The `actual_query_batch_bytes` and `actual_target_batch_bytes` specify the size of the two batches (in bytes) including the pad bases. `actual_n_alns` is the number of alignments to be performed. GASAL2 internally sets `is_free` to 0 after launching the alignment kernel on the GPU. From the performance prespective, if the average lengths of the sequences in *query_batch* and *target_batch* are not same, then the shorter sequences should be placed in *query_batch*. Forexample, in case of read mappers the read sequences are conatined in query_batch and the genome sequences in target_batch.


The `gasal_aln_async()` function returns immediately after launching the alignment kernel on the GPU. The user can perform other tasks instead of waiting for the kernel to finish. The output of alignments are stored in `host_aln_score`, `host_query_batch_end`, `host_target_batch_end`, `host_query_batch_start`, and `host_target_batch_start` arrays. To test whether the alignment on GPU is finished, a call to the following function is required to be made:

```
int gasal_is_aln_async_done(gasal_gpu_storage *gpu_storage);
```
If the function returns 0 the alignment on the GPU is finished and the output arrays contain valid results. Moreover, `is_free` is set to 1 by GASAL2. Thus, the current stream can be used for the alignment of another batch of sequences. 


## Problems and suggestions
For any issues and suugestions contact Nauman Ahmed (n.ahmed@tudelft.nl)

