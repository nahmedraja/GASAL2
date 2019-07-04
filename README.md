# GASAL2 - GPU-accelerated DNA alignment library
GASAL2 is an easy-to-use CUDA library for DNA/RNA sequence alignment algorithms. Currently it supports different kind of alignments:
 - local alignment
 - semi-global alignment
 - global alignment
 - tile-based banded alignment.
 
It can also reverse and, or complement any sequences independently before alignment, and report second-best scores for certain alignment types.

It is an extension of GASAL (https://github.com/nahmedraja/GASAL) and allows full overlapping of CPU and GPU execution. 

## List of new features:
- **Added traceback computation. The ouput is in CIGAR format**
- **GASAL2 can now compute all types of semi-global alignments**
- **Added expandable memory management on host side. The batches of query and target sequences are automatically enlarged if the required memory becomes larger than the allocated memory** 
- **Added kernel to reverse-complement sequences.**
- **Cleaned up, inconsistencies fixed, and a small optimization has been added (around 9% speedup with exact same result)** 


## List of changes:
- Changed the interface of `gasal_init_streams()` function
- The user now has to provide `MAX_QUERY_LEN` instead of `MAX_SEQ_LEN` during compilation

## Requirements
A Linux platform with CUDA toolkit 8 or higher is required, along with usual build environment for C and C++ code. GASAL2 has been tested over NVIDIA GPUs with compute capabilities of 2.0, 3.5 and 5.0. Although lower versions of the CUDA framework might work, they have not been tested.

## Compiling GASAL2
To compile the library, you need to specify the path of your CUDA installation and the variables for the Makefile in the script `run_all.sh`. Then you can compile GASAL2 by running this `run_all.sh` script. In the current script, an example of values is shown for  a GPU with Compute Capability of 3.5, a maximum sequence length of 300, a "N" code of 0xQF (representing the character "N"), and a N penalty of 1.

In this script, these are the two lines where you have to adjust the parameters:
```bash
$ ./configure.sh <path to cuda installation directory>
$ make GPU_SM_ARCH=<GPU SM architecture> MAX_QUERY_LEN=<maximum query length> N_CODE=<code for "N", e.g. 0x4E if the bases are represented by ASCII characters> [N_PENALTY=<penalty for aligning "N" against any other base>]
```

`N_PENALTY` is optional and if it is not specified then GASAL2 considers "N" as an ordinary base having the same match/mismatch scores as for A, C, G or T. As a result of these commands, *include* and *lib* directories will be created containing various .h files and `libgasal.a`, respectively. You will need to include part or all the .h files in your code link it with `libgasal.a` during compilation. Also link the CUDA runtime library by adding `-lcudart` flag. The path to the CUDA runtime library must also be specfied while linking as *-L <path to CUDA lib64 directory>*.

## Using GASAL2

### Initialization
To use GASAL2  alignment functions, first the match/mismatach scores and gap open/extension penalties need to be passed on to the GPU. Assign the values match/mismatach scores and gap open/extension penalties to the members of `gasal_subst_scores` struct:

```C
typedef struct{
	int32_t match;
	int32_t mismatch;
	int32_t gap_open;
	int32_t gap_extend;
}gasal_subst_scores;
```

The values are passed to the GPU by calling `gasal_copy_subst_scores()` function:

```C
void gasal_copy_subst_scores(gasal_subst_scores *subst);
```

A vector of `gasal_gpu_storage_t` is created a the following function:

```C
gasal_gpu_storage_v gasal_init_gpu_storage_v(int n_streams);
```

With the help of `n_streams`, the user specifies the number of outstanding GPU alignment kernel launches to be performed. The return type is `gasal_gpu_storage_v`:

```C
typedef struct{
	int n;
	gasal_gpu_storage_t *a;
}gasal_gpu_storage_v;
```

with `n = n_streams` and `a` being a pointer to the array. An element of the array holds the required data structurea of a stream. To destroy the vector the following function is used:

```C
void gasal_destroy_gpu_storage_v(gasal_gpu_storage_v *gpu_storage_vec);
```

The streams in the vector are initialized by calling:

```C
void gasal_init_streams(gasal_gpu_storage_v *gpu_storage_vec,  int max_query_len, int max_target_len, int max_n_alns,  Parameters *params);
```

In GASAL2, the sequences to be aligned are conatined in two batches. A sequence in query_batch is aligned to sequence in target_batch. A *batch* is a concatenation of sequences. *The length of a sequence must be a multiple of 8*. Hence, if a sequence is not a multiple of 8, `N's` are added at the end of sequence. We call these redundant bases as *Pad bases*. Note that the pad bases are always "N's" irrespective of whether `N_PENALTY` is defined or not. The `gasal_init_streams()` function alloctes the memory required by a stream. With the help of *max_batch_bytes*, the user specifies the expected maxumum size(in bytes) of sequences in the two batches. *host_max_batch_bytes* are pre-allocated on the CPU. Smilarly, *gpu_max_batch_bytes* are pre-allocated on the GPU. *max_n_alns* is the expected maximum number of sequences in a batch. If the actual required GPU memory is more than the pre-allocated memory, GASAL2 automatically allocates more memory. 

Most GASAL2 functions operate with a Parameters object. This object holds all the informations about the alignment options selected. In particular, the alignment type, the default values when opening or extending gaps, etc. The Parameters object is filled like this:

```C
Parameters *args;
args = new Parameters(0, NULL);

args->algo = <LOCAL|GLOBAL|SEMI_GLOBAL>; 
args->start_pos = <WITHOUT_START|WITH_START|WITH_TB>; //`WITHOUT_START` computes only the score and end-position. `WITH_START` computes the start-position with score and end-position. `WITH_TB` computes the score, start-position, end-position and traceback in CIGAR format.
args->isReverseComplement = <TRUE|FALSE>; //whether to reverse-complement the query sequence.
args->semiglobal_skipping_head = <QUERY|TARGET|BOTH|NONE>; //ignore gaps at the begining of QUERY|TARGET|BOTH|NONE in semi alignment-global.
args->semiglobal_skipping_tail = <QUERY|TARGET|BOTH|NONE>; //ignore gaps at the end of QUERY|TARGET|BOTH|NONE in semi alignment-global.
args->secondBest = <TRUE|FALSE>; //whether to compute the second best score in local and semi-global algo. But the start-position(WITH_START) and traceback(WITH_TRACEBACK) is only computed with the best score.

```


To free up the allocated memory the following function is used:

```C
void gasal_destroy_streams(gasal_gpu_storage_v *gpu_storage_vec, Parameters *params);
```

The `gasal_init_streams()` and `gasal_destroy_streams()` internally use `cudaMalloc()`, `cudaMallocHost()`, `cudaFree()` and `cudaFreeHost()` functions. These CUDA API functions are time expensive. Therefore, `gasal_init_streams()` and `gasal_destroy_streams()` should be preferably called only once in the program. You will find all these functions in the file `ctors.cpp`.


### Input data preparation
The `gasal_gpu_storage_t` in `gasal.h` holds the data structures for a stream. In the following we only show those members of `gasal_gpu_storage_t` which should be accessed by the user. Other fields should not be modified manually and the user should rely on dedicated functions for complex operations.

```C
typedef struct{
	...
	uint8_t *host_query_op;
	uint8_t *host_target_op;
	...
	uint32_t *host_query_batch_offsets;
	uint32_t *host_target_batch_offsets;
	uint32_t *host_query_batch_lens;
	uint32_t *host_target_batch_lens;
	uint32_t host_max_query_batch_bytes;
	uint32_t host_max_target_batch_bytes;
	gasal_res_t *host_res;
	gasal_res_t *host_res_second; 
	uint32_t host_max_n_alns;
	uint32_t current_n_alns;
	int is_free;
	...
} gasal_gpu_storage_t;
```



To align the sequences the user first need to check the availability of a stream. If `is_free` is  1, the user can use the current stream to perform the alignment on the GPU. 
To do this, the user must fill the sequences with the following function.

```C
uint32_t gasal_host_batch_fill(gasal_gpu_storage_t *gpu_storage, uint32_t idx, const char* data, uint32_t size, data_source SRC);

```

This function takes a sequence and its length, and append it in the data structure. It also adds the neccessary padding bases to ensure the sequence has a length which is a multiple of 8. Moreover, it takes care of allocating more memory if there is not enough room when adding the sequence. `SRC` is either `QUERY` or `TARGET`, depending upon which batch to fill. When executed, this function returns the offset to be filled by the user in `host_target_batch_offsets` or `host_query_batch_offsets`. The user also has to fill the length of sequences in `host_target_batch_lens` or `host_query_batch_lens`. The `current_n_alns` must appropriayely be incremented to show the current number of alignments. `host_max_n_alns` is initially set eequal to `max_n_alns` in `gasal_init_streams()` function. If the 'current_n_alns' exceeds `host_max_n_alns`, the user must call the following funnction to reallocate host offset, lengths and results arrays.

```C
void gasal_host_alns_resize(gasal_gpu_storage_t *gpu_storage, int new_max_alns, Parameters *params); 

```

where `new_max_alns` is the new value of `host_max_n_alns`.


One can also use the `gasal_host_batch_addbase` to add a single base to the sequence. This takes care of memory reallocation if needed, but does not take care of padding, so this has to be used carefully.


The the list of pre-processing operation (nothing, reverse, complement, reverse-complement) that has to be done on the batch of sequence can be loaded into the gpu_storage with the function `gasal_op_fill`. Its code is in `interfaces.cpp`. It fills `host_query_op` and `host_query_op` with an array of size `host_max_n_alns` where each value is the value of the enumeration of `operation_on_seq` (in gasal.h):
```C
enum operation_on_seq{
	FORWARD_NATURAL,
	REVERSE_NATURAL,
	FORWARD_COMPLEMENT,
	REVERSE_COMPLEMENT,
};
```
By default, no operations are done on the sequences (that is, the fields `host_query_op` and `host_target_op` arrays are initialized to 0, which is the value of FORWARD_NATURAL).


### Alignment launching
To launch the alignment, the following function is used:

```C
void gasal_aln_async(gasal_gpu_storage_t *gpu_storage, const uint32_t actual_query_batch_bytes, const uint32_t actual_target_batch_bytes, const uint32_t actual_n_alns, Parameters *params)
```

The `actual_query_batch_bytes` and `actual_target_batch_bytes` specify the size of the two batches (in bytes) including the pad bases. `actual_n_alns` is the number of alignments to be performed. GASAL2 internally sets `is_free` to 0 after launching the alignment kernel on the GPU. From the performance prespective, if the average lengths of the sequences in *query_batch* and *target_batch* are not same, then the shorter sequences should be placed in *query_batch*. Fo rexample, in case of read mappers the read sequences are conatined in query_batch and the genome sequences in target_batch.

The `gasal_aln_async()` function returns immediately after launching the alignment kernel on the GPU. The user can perform other tasks instead of waiting for the kernel to finish.To test whether the alignment on GPU is finished, the following function is called:

```
int gasal_is_aln_async_done(gasal_gpu_storage *gpu_storage);
```

If the function returns 0 the alignment on the GPU is finished and the output arrays contain valid results. Moreover, `is_free` is set to 1 by GASAL2. Thus, the current stream can be used for the alignment of another batch of sequences. The function returns `-1` if the results are not ready. It returns `-2` if the function is called on a stream in which no alignment has been launced, i.e. `is_free == 1`.


### Alignment results
The structure `gasal_res_t` holds the results of the alignment and can be accessed manually. Its fields are the following:

```C
struct gasal_res{
	int32_t *aln_score;
	int32_t *query_batch_end;
	int32_t *target_batch_end;
	int32_t *query_batch_start;
	int32_t *target_batch_start;
	uint8_t *cigar;
	uint32_t *n_cigar_ops;
};
typedef struct gasal_res gasal_res_t;
```
The output of alignments are stored in `aln_score`, `query_batch_end`, `target_batch_end`, `query_batch_start`, and `target_batch_start`, `cigar` and `n_cigar_ops` arrays, within the `host_res` structure inside the `gasal_gpu_storage` structure. `cigar` is a byte array which contains the traceback information in CIGAR format of all the alignments performed . The lower 2 bits of a byte indicate the CIGAR operation:

```
0 = match
1 = mismatch
2 = deletion
3 = insertion
```
The upper 6 bits store the count of the operation in the lower two bits. The traceback information of an alignment in the `cigar` array is in the reverse direction. `host_query_batch_offsets` conatins the offset of an alignment in the `cigar` array. The `n_cigar_ops` contains number of bytes in the cigar array encoding the traceback information of an alignment.

In case of second-best result, the same applies with the fields in `host_res_secondbest`. But the start-position and traceback( is only computed with the best score. Therefore, only `host_res_secondbest->aln_score`, `host_res_secondbest->query_batch_end` and `host_res_secondbest->target_batch_end` are valid for second-best result. 




## Example
The `test_prog` directory conatins an example program which uses GASAL2 for sequence alignment on GPU. See the README in the directory for the instructions about running the program.

## Problems and suggestions
For any issues and suugestions contact Jonathan Lévy (j.levy@student.tudelft.nl) or Nauman Ahmed (n.ahmed@tudelft.nl).





