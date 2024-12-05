#include <iostream>

//     |depth
//     |
//     |
//     |________width
//    /
//   /
//  /
// /height

#define KERNEL_RADIUS 1
__constant__ float const_conv_kernel_3D[(2 * KERNEL_RADIUS + 1) * (2 * KERNEL_RADIUS + 1) * (2 * KERNEL_RADIUS + 1)];

#define BLOCK_SIZE 16


void coutmatrix3D(const float* mat, int width, int height, int depth, const char* name){
    std::cout << "Matrix " << name << ":\n";
    for (int k = 0; k < depth; k++){
        std::cout << "Layer" << k << ":\n";
        for(int i = 0;i < height; i++){
            for(int j = 0; j < width; j++){
                std::cout << mat[k * height * width + i * width + j] << "\t";
            }
            std::cout << "\n";
        }
    }

}

__global__ void conv3D_basic_boundary_check(const float *A, const float *B, float *C, int width, int height, int depth, int r){
    int outDep = blockIdx.z * blockDim.z + threadIdx.z;
    int outHei = blockIdx.y * blockDim.y + threadIdx.y;
    int outWid = blockIdx.x * blockDim.x + threadIdx.x;
    if (outDep >= depth || outHei >= height || outWid >= width) return;
    float Pvalue = 0.0f;
    for (int fDep = 0; fDep < 2 * r + 1; fDep++){
        for (int fHei = 0; fHei < 2 * r + 1; fHei++){
            for (int fWid = 0; fWid < 2 * r + 1;fWid++){
                int inDep = outDep - r + fDep;
                int inHei = outHei - r + fHei;
                int inWid = outWid - r + fWid;
                if (inDep >= 0 && inHei >= 0 && inWid >= 0 && inDep < depth && inHei < height && inWid < width){
                    Pvalue += A[inDep * height * width + inHei * width + inWid] * B[fDep * (2 * r + 1) * (2 * r + 1) + fHei * (2 * r + 1) + fWid];
                }
            }
        }
    }
    C[outDep * height * width + outHei * width + outWid] = Pvalue;
}

__global__ void conv3D_constant_mem(float *A, float *C, int width, int height, int depth){
    int outDep = blockDim.z * blockIdx.z + threadIdx.z;
    int outHei = blockDim.y * blockIdx.y + threadIdx.y;
    int outWid = blockDim.x * blockIdx.x + threadIdx.x;
    if (outDep >= depth || outHei >= height || outWid >= width) return;
    float Pvalue = 0.0f;
    for (int fDep = 0; fDep < 2 * KERNEL_RADIUS + 1; fDep++){
        for (int fHei = 0; fHei < 2 * KERNEL_RADIUS + 1; fHei++){
            for (int fWid = 0; fWid < 2 * KERNEL_RADIUS + 1;fWid++){
                int inDep = outDep - KERNEL_RADIUS + fDep;
                int inHei = outHei - KERNEL_RADIUS + fHei;
                int inWid = outWid - KERNEL_RADIUS + fWid;
                if (inDep >= 0 && inHei >= 0 && inWid >= 0 && inDep < depth && inHei < height && inWid < width){
                    Pvalue += A[inDep * height * width + inHei * width + inWid] * const_conv_kernel_3D[fDep * (2 * KERNEL_RADIUS + 1) * (2 * KERNEL_RADIUS + 1) + fHei * (2 * KERNEL_RADIUS + 1) + fWid];
                }
            }
        }
    }
    C[outDep * height * width + outHei * width + outWid] = Pvalue;
}

__global__ void conv3D_constant_mem_tiled(const float *A, float *C, int width, int height, int depth){
    __shared__ float sharedMem[BLOCK_SIZE + 2 * KERNEL_RADIUS][BLOCK_SIZE + 2 * KERNEL_RADIUS][BLOCK_SIZE + 2 * KERNEL_RADIUS];
    int outDep = blockDim.z * blockIdx.z + threadIdx.z;
    int outHei = blockDim.y * blockIdx.y + threadIdx.y;
    int outWid = blockDim.x * blockIdx.x + threadIdx.x;
    
    for (int z = threadIdx.z; z < BLOCK_SIZE + 2 * KERNEL_RADIUS; z += BLOCK_SIZE){
        int globalDep = blockIdx.z * blockDim.z + z - KERNEL_RADIUS;
        for (int y = threadIdx.y; y < BLOCK_SIZE + 2 * KERNEL_RADIUS; y += BLOCK_SIZE){
            int globalHei = blockIdx.y * blockDim.y + y - KERNEL_RADIUS;
            for (int x = threadIdx.x; x < BLOCK_SIZE + 2 * KERNEL_RADIUS; x += BLOCK_SIZE){
                int globalWid = blockIdx.x * blockDim.x + x - KERNEL_RADIUS;
                if (globalDep >=0 && globalHei >= 0 && globalWid >= 0 && globalDep < depth && globalHei < height && globalWid < width){
                    sharedMem[z][y][x] = A[globalDep * height * width + globalHei * width + globalWid];
                } else{
                    sharedMem[z][y][x] = 0.0f;
                }
            }
        }
    }
    __syncthreads();

    if (outDep >= depth || outHei >= height || outWid >= width) return;
    float Pvalue = 0.0f;
    for (int i = 0; i < 2 * KERNEL_RADIUS + 1; i++){
        for (int j = 0; j < 2 * KERNEL_RADIUS + 1; j++){
            for (int k = 0; k < 2 * KERNEL_RADIUS + 1; k++){
                Pvalue += sharedMem[i + threadIdx.z][j + threadIdx.y][k + threadIdx.x] * const_conv_kernel_3D[i * (2 * KERNEL_RADIUS + 1) * (2 * KERNEL_RADIUS + 1) + j * (2 * KERNEL_RADIUS + 1) + k];
            }
        }
    }
    C[outDep * height * width + outHei * width + outWid] = Pvalue;
}

void conv3D(const float *A, const float *B, float *C, int width, int height, int depth, int r){
    float *d_A, *d_B, *d_C;
    size_t size_A = width * height * depth * sizeof(float);
    size_t size_B = (2 * r + 1) * (2 * r + 1) * (2 * r + 1) * sizeof(float);

    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_A);//size C is the same as the size A

    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    dim3 blockDim(8,8,8);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, (depth + blockDim.z - 1) / blockDim.z);
    conv3D_basic_boundary_check<<<gridDim, blockDim>>>(d_A, d_B, d_C, width, height, depth, r);
    cudaMemcpy(C, d_C, size_A, cudaMemcpyDeviceToHost);
    coutmatrix3D(C, width, height, depth, "C_3D");

    cudaFree(d_C);
    cudaMemcpyToSymbol(const_conv_kernel_3D, B, size_B);
    conv3D_constant_mem<<<gridDim, blockDim>>>(d_A, d_C, width, height, depth);
    cudaMemcpy(C, d_C, size_A, cudaMemcpyDeviceToHost);
    coutmatrix3D(C, width, height, depth, "C_3D_constant_mem");

    cudaFree(d_C);
    conv3D_constant_mem_tiled<<<gridDim, blockDim>>>(d_A, d_C, width, height, depth);
    cudaMemcpy(C, d_C, size_A, cudaMemcpyDeviceToHost);
    coutmatrix3D(C, width, height, depth, "C_3D_constant_mem_tile");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main(){
    const int width_A = 5;
    const int heigh_A = 5;
    const int depth_A = 5;
    const int r = 1;
    float h_A_layer[] = {
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
        1.0f, 2.0f, 2.0f, 2.0f, 1.0f, 
        1.0f, 2.0f, 3.0f, 2.0f, 1.0f, 
        1.0f, 2.0f, 2.0f, 2.0f, 1.0f, 
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f 
    };
    const int size_A = width_A * heigh_A * depth_A;
    float h_A[size_A];
    for (int d = 0; d < depth_A; d++){
        for (int i = 0;i < width_A * heigh_A; i++){
            h_A[d * heigh_A * width_A + i] = h_A_layer[i];
        }
    }

    // float h_B_layer[] = {
    //     0, 0, 0,
    //     0, 5, 0,
    //     0, 0, 0
    // };
    // const int size_B = (2 * r + 1) * (2 * r + 1) * (2 * r + 1);
    // float h_B[size_B];
    // for (int d = 0; d < 2 * r + 1; d++){
    //     for (int i = 0;i < (2 * r + 1) * (2 * r + 1);i++){
    //         h_B[d * (2 * r + 1) * (2 * r + 1) + i] = h_B_layer[i];
    //     }
    // }

    const int size_B = (2 * r + 1) * (2 * r + 1) * (2 * r + 1);
    float h_B[size_B];
    for (int i = 0; i < size_B; i++) {
        h_B[i] = 0;
    }
    h_B[(size_B - 1) / 2] = 5;

    
    coutmatrix3D(h_A, width_A, heigh_A, depth_A, "A_3D");
    coutmatrix3D(h_B, 2*r + 1, 2 * r + 1, 2 * r + 1, "conv_kernel_3D");

    // float* h_C = new float[row_A * col_A];//same size as A
    float h_C[size_A];
    conv3D(h_A, h_B, h_C, width_A, heigh_A, depth_A, r);

    
    // delete[] h_C;
    return 0;
}