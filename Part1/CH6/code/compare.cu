#include <iostream>
#include <cuda_runtime.h>
#include <cstring> // For memcpy

#define TILE_SIZE 16

// Naive Matrix Multiplication Kernel
__global__ void MatMulNaive(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int row_A, int col_A, int col_B){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < row_A && col < col_B){
        float Cvalue = 0.0f;
        for(int k = 0; k < col_A; k++){
            Cvalue += A[row * col_A + k] * B[k * col_B + col];
        }
        C[row * col_B + col] = Cvalue;
    }
}

// Shared Memory Tiled Matrix Multiplication Kernel
__global__ void MatMulCornerTurn(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int row_A, int col_A, int col_B){
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float Cvalue = 0.0f;

    for (int t = 0; t < (col_A + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load shared memory for A
        if (row < row_A && (t * TILE_SIZE + threadIdx.x) < col_A) {
            As[threadIdx.y][threadIdx.x] = A[row * col_A + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load shared memory for B, col_A = row_B
        if ((t * TILE_SIZE + threadIdx.y) < col_A && col < col_B) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * col_B + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Perform multiplication for the tile
        for (int i = 0; i < TILE_SIZE; i++) {
            Cvalue += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the result to C
    if (row < row_A && col < col_B) {
        C[row * col_B + col] = Cvalue;
    }
}

// Host Function for Matrix Multiplication
void matrixMultiply(const float* A, const float* B, float* C, int row_A, int col_A, int col_B, bool useSharedMemory){
    float *d_A, *d_B, *d_C;
    size_t size_A = row_A * col_A * sizeof(float);
    size_t size_B = col_A * col_B * sizeof(float);
    size_t size_C = row_A * col_B * sizeof(float);

    // Allocate device memory
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    // Copy data from host to device
    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((col_B + TILE_SIZE - 1) / TILE_SIZE, (row_A + TILE_SIZE - 1) / TILE_SIZE);

    // Select kernel based on the flag
    if(useSharedMemory){
        MatMulCornerTurn<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, row_A, col_A, col_B);
    }
    else{
        MatMulNaive<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, row_A, col_A, col_B);
    }

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Copy the result from device to host
    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Function to Print Matrices
void coutmatrix(const float* mat, int rows, int cols, const char* name){
    std::cout << "Matrix " << name << ":\n";
    for(int i = 0;i < rows; i++){
        for(int j = 0; j < cols; j++){
            std::cout << mat[i * cols + j] << "\t";
        }
        std::cout << "\n";
    }
}

// Function to Initialize Matrices
void initializeMatrices(float* h_A, float* h_B, int row_A, int col_A, int col_B){
    // Initialize matrix A
    // Example: Initialize A with repeating sequences
    for(int i = 0; i < row_A; i++){
        for(int j = 0; j < col_A; j++){
            h_A[i * col_A + j] = static_cast<float>((i * col_A + j) % 6 + 1) * 1.012; // Values between 1.0f and 6.0f
        }
    }

    // Initialize matrix B
    // Example: Initialize B with alternating 1s and 2s
    for(int i = 0; i < col_A; i++){
        for(int j = 0; j < col_B; j++){
            h_B[i * col_B + j] = (j % 2 == 0) ? 1.0f : 2.0f;
            h_B[i * col_B + j] *= 1.023;
        }
    }
}

int main(){
    // Define matrix dimensions
    int row_A = 1024*2; // Increased size for meaningful performance comparison
    int col_A = 1024*2;
    int col_B = 1024*2;

    // Allocate host memory
    float *h_A = (float*)malloc(row_A * col_A * sizeof(float));
    float *h_B = (float*)malloc(col_A * col_B * sizeof(float));
    float *h_C_naive = (float*)malloc(row_A * col_B * sizeof(float));
    float *h_C_shared = (float*)malloc(row_A * col_B * sizeof(float));

    // Initialize matrices
    initializeMatrices(h_A, h_B, row_A, col_A, col_B);

    // Uncomment the following lines if you want to print the matrices (Not recommended for large matrices)
    /*
    coutmatrix(h_A, row_A, col_A, "A");
    coutmatrix(h_B, col_A, col_B, "B");
    */

    // Create CUDA events for timing
    cudaEvent_t start_naive, stop_naive;
    cudaEvent_t start_shared, stop_shared;
    cudaEventCreate(&start_naive);
    cudaEventCreate(&stop_naive);
    cudaEventCreate(&start_shared);
    cudaEventCreate(&stop_shared);

    // --------------------------
    // Measure Naive Kernel
    // --------------------------
    cudaEventRecord(start_naive);
    matrixMultiply(h_A, h_B, h_C_naive, row_A, col_A, col_B, false);
    cudaEventRecord(stop_naive);
    cudaEventSynchronize(stop_naive);

    float milliseconds_naive = 0;
    cudaEventElapsedTime(&milliseconds_naive, start_naive, stop_naive);

    // --------------------------
    // Measure Shared Memory Kernel
    // --------------------------
    cudaEventRecord(start_shared);
    matrixMultiply(h_A, h_B, h_C_shared, row_A, col_A, col_B, true);
    cudaEventRecord(stop_shared);
    cudaEventSynchronize(stop_shared);

    float milliseconds_shared = 0;
    cudaEventElapsedTime(&milliseconds_shared, start_shared, stop_shared);

    // --------------------------
    // Compare Results (Optional)
    // --------------------------
    bool correct = true;
    for(int i = 0; i < row_A * col_B; i++){
        if(abs(h_C_naive[i] - h_C_shared[i]) > 1e-3){
            correct = false;
            std::cout << "Mismatch at index " << i << ": Naive = " << h_C_naive[i] << ", Shared = " << h_C_shared[i] << "\n";
            break;
        }
    }

    if(correct){
        std::cout << "Both kernels produced the same results.\n";
    }
    else{
        std::cout << "Mismatch detected between kernel results!\n";
    }

    // --------------------------
    // Print Execution Times
    // --------------------------
    std::cout << "Naive Kernel Execution Time: " << milliseconds_naive << " ms\n";
    std::cout << "Shared Memory Kernel Execution Time: " << milliseconds_shared << " ms\n";

    // --------------------------
    // Clean Up
    // --------------------------
    free(h_A);
    free(h_B);
    free(h_C_naive);
    free(h_C_shared);

    cudaEventDestroy(start_naive);
    cudaEventDestroy(stop_naive);
    cudaEventDestroy(start_shared);
    cudaEventDestroy(stop_shared);

    return 0;
}
