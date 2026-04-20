
#define KERNELS_H
__global__ void negate_kernel(const int id, const int* input, const int size, int* output);
__global__ void dot_product(const int A_ROWS, const int A_COLS, const int B_ROWS, const int B_COLS, const float* A, const float* B, float* result);
__global__ void dot_product_tiled(const int A_ROWS, const int A_COLS, const int B_ROWS, const int B_COLS, const int TILE_DIM, const float* A, const float* B, float* result);
