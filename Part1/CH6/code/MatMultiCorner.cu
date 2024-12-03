#include<iostream>

#define tile_size 16

__global__ void MatMulCornerTurn(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int row_A, int col_A, int col_B){
    int row = blockIdx.y * tile_size + threadIdx.y;
    int col = blockIdx.x * tile_size + threadIdx.x;

    __shared__ float As[tile_size][tile_size];
    __shared__ float Bs[tile_size][tile_size];

    float Cvalue = 0.0f;

    for (int t = 0; t < (col_A + tile_size - 1) / tile_size; t++) {
        // Load shared memory for A
        if (row < row_A && (t * tile_size + threadIdx.x) < col_A) {
            As[threadIdx.y][threadIdx.x] = A[row * col_A + t * tile_size + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load shared memory for B, col_A = row_B
        if ((t * tile_size + threadIdx.y) < col_A && col < col_B) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * tile_size + threadIdx.y) * col_B + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Perform multiplication for the tile
        for (int i = 0; i < tile_size; i++) {
            Cvalue += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the result to C
    if (row < row_A && col < col_B) {
        C[row * col_B + col] = Cvalue;
    }
}


void matrixMultiply(const float* A, const float* B, float* C, int row_A, int col_A, int col_B){
    float *d_A, *d_B, *d_C;
    size_t size_A = row_A * col_A * sizeof(float);
    size_t size_B = col_A * col_B * sizeof(float);
    size_t size_C = row_A * col_B * sizeof(float);

    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    dim3 dimBlock(tile_size, tile_size);
    dim3 dimGrid((col_B + tile_size - 1) / tile_size, (row_A + tile_size - 1) / tile_size);

    MatMulCornerTurn<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, row_A, col_A, col_B);

    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

//print matrix
void coutmatrix(const float* mat, int rows, int cols, const char* name){
    std::cout << "Matrix " << name << ":\n";
    for(int i = 0;i < rows; i++){
        for(int j = 0; j < cols; j++){
            std::cout << mat[i * cols + j] << "\t";
        }
        std::cout << "\n";
    }
}

int main(){
    int row_A = 18;//also row_C
    int col_A = 6;//also row_B
    int col_B = 4;//also col_C
    float *h_A = (float*)malloc(row_A * col_A * sizeof(float));
    float *h_B = (float*)malloc(col_A * col_B * sizeof(float));
    float *h_C = (float*)malloc(row_A * col_B * sizeof(float));

    float host_A[] = {1.0f, 2.0f, 3.0f,  1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f, 4.0f, 5.0f, 6.0f,
                        1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f, 4.0f, 5.0f, 6.0f,
                        1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f, 4.0f, 5.0f, 6.0f,
                        1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f, 4.0f, 5.0f, 6.0f,
                        1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f, 4.0f, 5.0f, 6.0f,
                        1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f, 4.0f, 5.0f, 6.0f,
                        1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f, 4.0f, 5.0f, 6.0f,
                        1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f, 4.0f, 5.0f, 6.0f,
                        1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f, 4.0f, 5.0f, 6.0f,};

    float host_B[] = {1.0f, 2.0f,1.0f, 2.0f,
                    1.0f, 2.0f,  1.0f, 2.0f,
                    1.0f, 2.0f,  1.0f, 2.0f,
                    1.0f, 2.0f,  1.0f, 2.0f,
                    1.0f, 2.0f,  1.0f, 2.0f,
                    1.0f, 2.0f,  1.0f, 2.0f};
    memcpy(h_A, host_A, sizeof(host_A));
    memcpy(h_B, host_B, sizeof(host_B));
    
    coutmatrix(h_A, row_A, col_A, "A");
    coutmatrix(h_B, col_A, col_B, "B");
    // constexpr int tile_size = 16;
    matrixMultiply(h_A, h_B, h_C, row_A, col_A, col_B);
    coutmatrix(h_C, row_A, col_B, "C");

    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}