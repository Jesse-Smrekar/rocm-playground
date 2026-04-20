#include <hip/hip_runtime.h>


__global__ void negate_kernel(	const int id,
								const int* input,
								const int arrSize,
								int* output) {
	// in parallel, get (arrSize) ints starting at the input pointer location
	// multilpy the values by -1 and then store them in the same order at the output array
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	printf("Kernel ID: %d\n", id);
	printf("   processing %d\n", idx);

	((int*)output)[idx] = input[idx] * -1;
}



/** function that calcualtes the dot product of two (float) matricies (passed as float*)
 * 	https://www.youtube.com/watch?v=Q3GgbfGTnVc
 * 
 * A_ROWS 	-	the width of matrix A
 * A_COLS	-	the height of matrix A
 * B_ROWS	-	the width of matrix B
 * B_COLS	-	the height of matrix B
 * A		- 	pointer to the A matrix data
 * B		- 	pointer to the B matrix data
 * result	- 	pointer to resulting matrix
 */

__global__ void dot_product(const int A_ROWS, const int A_COLS, const int B_ROWS, const int B_COLS,
							const float *A, const float *B, float *result) {

	int target = (blockIdx.x * blockDim.x) + threadIdx.x;

	// printf("I'm blockId: %d, threadId: %d - and the block dim: %d \n", blockIdx.x, threadIdx.x, blockDim.x);
	if (target > A_ROWS * B_COLS) {
		return;
	}

	int aPos = (target / A_COLS) * A_COLS; // looks redundant but gets you to the start of the row
	int bPos = target % B_COLS;
	int workingSum = 0;

	for (int i=0; i<A_ROWS; i++) {
		workingSum += A[aPos + i] * B[bPos * (i + 1)];
	}
	
	result[target] = workingSum;
	
	return;
}









/** function that calcualtes the dot product of two (float) matricies (passed as float*) using a tiled approach
 * for shared memory efficiency.
 * 	https://www.youtube.com/watch?v=Q3GgbfGTnVc
 * 
 * A_ROWS 	-	the width of matrix A
 * A_COLS	-	the height of matrix A
 * B_ROWS	-	the width of matrix B
 * B_COLS	-	the height of matrix B
 * TILE_DIM	-	the dimension of the square tile (subset of the matricies) we are wanting to process in one thread
 * A		- 	pointer to the A matrix data
 * B		- 	pointer to the B matrix data
 * result	- 	pointer to resulting matrix
 */

__global__ void dot_product_tiled(const int A_ROWS, const int A_COLS, const int B_ROWS, const int B_COLS,
							const int TILE_DIM,
							const float *A, const float *B, float *result) {

	int target = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (target > A_ROWS * B_COLS) {
		return;
	}


	// allocate shared memory space for working tile
	// __shared__ float workingTile[A_COLS + B_ROWS];



}