#include <cuda_runtime.h>
#include <stdio.h>

// Matrix dimensions
#define M 1024 // Number of rows in A and C
#define N 1024 // Number of columns in A and rows in B
#define P 1024 // Number of columns in B and C

// CUDA Kernel: Each thread computes one row of the output matrix C
__global__ void matrixMulRowKernel(const float* A, const float* B, float* C, int N, int P) {
    // Calculate the row index this thread is responsible for
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check to ensure we don't access out-of-bounds memory
    if (row < M) {
        for (int col = 0; col < P; ++col) {
            float value = 0.0f;
            for (int k = 0; k < N; ++k) {
                value += A[row * N + k] * B[k * P + col];
            }
            C[row * P + col] = value;
        }
    }
}
