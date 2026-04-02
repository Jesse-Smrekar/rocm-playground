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