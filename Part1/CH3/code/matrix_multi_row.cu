#include <iostream>
#include "CudaMatrix.h"
// |1 2 3| * |1 2| = |6  12|
// |4 5 6|   |1 2|   |15 30|
//           |1 2|
__global__ void matrix_multi_row(const float* A, const float* B, float* C,int A_rows, int A_cols, int B_cols){
    //count in row order
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < A_rows){
        for(int col = 0; col < B_cols; col++){
            float sum = 0;
            for(int mul = 0; mul < A_cols; mul++){
                sum += A[row * A_cols + mul] * B[mul * B_cols + col];
            }
            C[row * B_cols + col] = sum;
        }
    }
}

__global__ void matrix_multi_col(const float* A, const float* B, float* C, int A_rows, int A_cols, int B_cols){
    //count in col order
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < B_cols){
        for(int row = 0; row < A_rows; row++){
            float sum = 0;
            for(int mul = 0; mul < A_cols; mul++){
                sum += A[row * A_cols + mul] * B[mul * B_cols + col];
            }
            C[row * B_cols + col] = sum;
        }
    }
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
    const int A_ROWS = 2;
    const int B_COLS = 2;
    float h_A[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float h_B[] = {1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f};
    const int A_COLS = sizeof(h_A) / sizeof(float) / A_ROWS;
    const int B_ROWS = sizeof(h_B) / sizeof(float) / B_COLS;
    // std::cout << COLS << std::endl; //output 3, correct
    size_t A_size = A_ROWS * A_COLS * sizeof(float);
    size_t B_size = B_ROWS * B_COLS * sizeof(float);
    size_t C_size = A_ROWS * B_COLS * sizeof(float);
    float h_C_row[A_ROWS * B_COLS];
    float h_C_col[A_ROWS * B_COLS];

    coutmatrix(h_A, A_ROWS, A_COLS, "A");
    coutmatrix(h_B, B_ROWS, B_COLS, "B"); // row_A = col_B

    
    float *d_A, *d_B, *d_C_col, *d_C_row;
    cudaMalloc((void**)&d_A, A_size);
    cudaMalloc((void**)&d_B, B_size);
    cudaMalloc((void**)&d_C_col, C_size);
    cudaMalloc((void**)&d_C_row, C_size);

    cudaMemcpy(d_A, h_A, A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, B_size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;

    // calculate in row:
    int blocksPerGrid = (A_ROWS + threadsPerBlock - 1) / threadsPerBlock;
    matrix_multi_row<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C_row, A_ROWS, A_COLS, B_COLS);

    // calculate in col:
    blocksPerGrid = (A_COLS + threadsPerBlock - 1) / threadsPerBlock;
    matrix_multi_col<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C_col, A_ROWS, A_COLS, B_COLS);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){//check if error happened;
        std::cerr << "Cuda launch error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C_row);
        cudaFree(d_C_col);
        return -1;
    }

    cudaDeviceSynchronize();
    cudaMemcpy(h_C_row, d_C_row, C_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_col, d_C_col, C_size, cudaMemcpyDeviceToHost);
    coutmatrix(h_C_row, A_ROWS, B_COLS, "C = A * B in row");
    coutmatrix(h_C_col, A_ROWS, B_COLS, "C = A * B in col");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_row);
    cudaFree(d_C_col);

    return 0;
}