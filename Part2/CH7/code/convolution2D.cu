#include <iostream>
//print matrix

#define MAX_KERNEL_RADIUS 1
__constant__ float const_conv_kernel[(2 * MAX_KERNEL_RADIUS + 1) * (2 * MAX_KERNEL_RADIUS + 1)];

#define BLOCK_SIZE 16


void coutmatrix(const float* mat, int rows, int cols, const char* name){
    std::cout << "Matrix " << name << ":\n";
    for(int i = 0;i < rows; i++){
        for(int j = 0; j < cols; j++){
            std::cout << mat[i * cols + j] << "\t";
        }
        std::cout << "\n";
    }
}
__global__ void conv2D_basic_boundary_check(const float *A, const float *B, float *C, int row, int col, int r){
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    if (outCol >= col || outRow >= row) return;
    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < 2 * r + 1; fRow++){
        for (int fCol = 0; fCol < 2 * r + 1; fCol++){
            int inRow = outRow - r + fRow;
            int inCol = outCol - r + fCol;
            if (inRow >= 0 && inRow < row && inCol >= 0 && inCol < col){
                Pvalue += B[fRow* (2 * r + 1) + fCol] * A[inRow * col + inCol];
            }
        }
    }
    C[outRow * col + outCol] = Pvalue;
}
__global__ void conv2D_basic(const float *A, const float *B, float *C, int row, int col, int r){
    //without boundary check, thus it will be smaller, row and col reduced by 2 * r;
    int outCol = blockDim.x * blockIdx.x + threadIdx.x;
    int outRow = blockDim.y * blockIdx.y + threadIdx.y;
    if (outCol < r || outCol >= col - r || outRow < r || outRow >= row - r) return;
    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < 2 * r + 1; fRow++){
        for (int fCol = 0; fCol < 2 * r + 1; fCol++){
            int inRow = outRow - r + fRow;
            int inCol = outCol - r + fCol;
            Pvalue += B[fRow * (2 * r + 1) + fCol] * A[inRow * col + inCol];
        }
    }
    C[(outRow - r) * (col - 2 * r) + (outCol - r)] = Pvalue;
}
__global__ void conv2D_constant_mem(const float *A, float *C, int row, int col){
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    if (outCol >= col || outRow >= row) return;
    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < 2 * MAX_KERNEL_RADIUS + 1; fRow++){
        for (int fCol = 0; fCol < 2 * MAX_KERNEL_RADIUS + 1; fCol++){\
            int inRow = outRow - MAX_KERNEL_RADIUS + fRow;
            int inCol = outCol - MAX_KERNEL_RADIUS + fCol;
            if (inRow >= 0 && inRow < row && inCol >= 0 && inCol < col){
                Pvalue += const_conv_kernel[fRow * (2 * MAX_KERNEL_RADIUS + 1) + fCol] * A[inRow * col + inCol];
            }
        }
    }
    C[outRow * col + outCol] = Pvalue;
}

__global__ void conv2D_constant_mem_tiled(const float *A, float *C, int row, int col){
    __shared__ float sharedMem[BLOCK_SIZE + 2 * MAX_KERNEL_RADIUS][BLOCK_SIZE + 2 * MAX_KERNEL_RADIUS];
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;

    for (int y = threadIdx.y; y < BLOCK_SIZE + 2 * MAX_KERNEL_RADIUS; y += BLOCK_SIZE){
        for (int x = threadIdx.x; x < BLOCK_SIZE + 2 * MAX_KERNEL_RADIUS; x += BLOCK_SIZE){
            int sharedMemRow = y;
            int sharedMemCol = x;
            int globalRow = blockIdx.y * blockDim.y + y - MAX_KERNEL_RADIUS;
            int globalCol = blockIdx.x * blockDim.x + x - MAX_KERNEL_RADIUS;

            if (globalRow >= 0 && globalRow < row && globalCol >= 0 && globalCol < col){
                sharedMem[sharedMemRow][sharedMemCol] = A[globalRow * col + globalCol];
            } else{
                sharedMem[sharedMemRow][sharedMemCol] = 0.0f;
            }
        }
    }

    __syncthreads();

    if (outRow < row && outCol < col){
        float Pvalue = 0.0f;
        for (int i = 0; i < 2 * MAX_KERNEL_RADIUS + 1; i ++){
            for (int j = 0; j < 2 * MAX_KERNEL_RADIUS + 1; j++){
                Pvalue += const_conv_kernel[i * (2 * MAX_KERNEL_RADIUS + 1) + j] * sharedMem[threadIdx.y + i][threadIdx.x + j];   
            }
        }
        C[outRow * col + outCol] = Pvalue;
    }
}

void conv2D(const float *A, const float *B, float *C, int row, int col, int r){
    float *d_A, *d_B, *d_C;
    size_t size_A = row * col * sizeof(float);
    size_t size_B = (2 * r + 1) * (2 * r + 1) * sizeof(float);

    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_A);//size C is the same as the size A

    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    dim3 blockDim(16,16);
    dim3 gridDim((col + blockDim.x - 1) / blockDim.x, (row + blockDim.y - 1) / blockDim.y);

    conv2D_basic_boundary_check<<<gridDim, blockDim>>>(d_A, d_B, d_C, row, col, r);
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, size_A, cudaMemcpyDeviceToHost);
    coutmatrix(C, row, col, "Result_boundary");

    size_t size_C = (row - r * 2) * (col - r * 2) * sizeof(float);
    cudaFree(d_C);
    cudaMalloc((void**)&d_C, size_C);//smaller than the size A with r
    dim3 grid((col - 2 * r + blockDim.x - 1) / blockDim.x,
                 (row - 2 * r + blockDim.y - 1) / blockDim.y);
    conv2D_basic<<<grid, blockDim>>>(d_A, d_B, d_C, row, col, r);
    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);
    coutmatrix(C, row - r * 2, col - r * 2, "Result_no_boundary");

    cudaMemcpyToSymbol(const_conv_kernel, B, size_B);
    cudaFree(d_C);
    cudaMalloc((void**)&d_C, size_A);
    conv2D_constant_mem<<<gridDim, blockDim>>>(d_A, d_C, row, col);
    cudaMemcpy(C, d_C, size_A, cudaMemcpyDeviceToHost);
    coutmatrix(C, row, col, "Constant_memory_res");
    
    cudaFree(d_C);
    cudaMalloc((void**)&d_C, size_A);
    conv2D_constant_mem_tiled<<<gridDim, blockDim>>>(d_A, d_C, row, col);
    cudaMemcpy(C, d_C, size_A, cudaMemcpyDeviceToHost);
    coutmatrix(C, row, col, "Constant_tiled_memory_res");
 


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main(){
    const int row_A = 5;
    const int col_A = 5;
    const int r = 1;
    float h_A[] = {
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
        1.0f, 2.0f, 2.0f, 2.0f, 1.0f, 
        1.0f, 2.0f, 3.0f, 2.0f, 1.0f, 
        1.0f, 2.0f, 2.0f, 2.0f, 1.0f, 
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f 
    };
    float h_B[] = {
        0, 0, 0,
        0, 5, 0,
        0, 0, 0
    };
    // std::cout<< r/2 * 2 << std::endl;
    coutmatrix(h_A, row_A, col_A, "A");
    coutmatrix(h_B, 2*r + 1, 2 * r + 1, "conv_kernel");

    float* h_C = new float[row_A * col_A];//same size as A
    conv2D(h_A, h_B, h_C, row_A, col_A, r);

    
    delete[] h_C;
    return 0;
}