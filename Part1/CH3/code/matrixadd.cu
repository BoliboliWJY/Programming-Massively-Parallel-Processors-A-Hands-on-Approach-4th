#include <cuda_runtime.h>
#include <iostream>

// #define ROWS 2
// #define COLS 2
//matrix add together
__global__ void matrixadd(const float* A, const float* B, float* C,int rows, int cols){
    //the pointers of the input matrics A and B, the pointer of the output matrix C, number of columns in matrices
    int row = blockIdx.x * blockDim.x + threadIdx.x;//global row index
    if (row < rows){
        for (int col = 0; col < cols; col++){//iterate each row
            C[row * cols + col] = A[row * cols + col] + B[row * cols + col];
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


int main() {
    const int ROWS = 2;
    const int COLS = 2;
    size_t size = ROWS * COLS * sizeof(float);
    //initalize matrix A,B,C in host
    float h_A[ROWS * COLS] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_B[ROWS * COLS] = {1.0f, 1.0f, 2.0f, 2.0f};
    float h_C[ROWS * COLS];

    // Print input matrices
    coutmatrix(h_A, ROWS, COLS, "A");
    coutmatrix(h_B, ROWS, COLS, "B");

    // set device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    //cpoy host data to device
    // cudaMemcpy(destination, source, size, direction);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    int threadsPerBlock = 256;
    int blocksPerGrid = (ROWS + threadsPerBlock - 1) / threadsPerBlock;//an extra block for not perfectly divisible by threadsPerBlock;
    matrixadd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, ROWS, COLS);//activate kernel, <<<...>>>specifying the number of grid and block;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){//check if error happened;
        std::cerr << "Cuda launch error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return -1;
    }

    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    coutmatrix(h_C, ROWS, COLS, "C=A+B");
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}
