#include <iostream>

void coutarray(const float* arr, int len, const char* name){
    std::cout << "Array " << name << ":\n";
    for (int i = 0; i < len; i++){
        std::cout << arr[i] << "\t";
    }
    std::cout << "\n";
}

#define SECTION_SIZE 256

__global__ void Kogge_Stone_scan_kernel(const float *X, float *Y, const int len){
    __shared__ float XY[SECTION_SIZE];
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < len){
        XY[threadIdx.x] = X[i];
    } else{
        XY[threadIdx.x] = 0.0f;
    }
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2){
        __syncthreads();
        float tmp;
        if(threadIdx.x >= stride){
            tmp = XY[threadIdx.x] + XY[threadIdx.x - stride];
            XY[threadIdx.x] = tmp;
        }
        __syncthreads();
    }
    if (i < len){
        Y[i] = XY[threadIdx.x];
    }
}

__global__ void Kogge_Stone_scan_kernel_double_buffer(const float *X, float *Y, const int len){
    __shared__ float XY[SECTION_SIZE];
    __shared__ float XY2[SECTION_SIZE];

    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    if ( i < len){
        XY[threadIdx.x] = X[i];
    } else{
        XY[threadIdx.x] = 0.0f;
    }
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2){
        __syncthreads();
        if(threadIdx.x >= stride){
            float tmp = XY[threadIdx.x] + XY[threadIdx.x - stride];
            XY2[threadIdx.x] = tmp;
        } else{
            XY2[threadIdx.x] = XY[threadIdx.x];
        }
        __syncthreads();
        for (int j = 0; j < blockDim.x; j++){
            XY[j] = XY2[j];
        }
    }
    __syncthreads();
        if (i < len+1){
            Y[i] = XY[threadIdx.x];
        }
}

__global__ void Kogge_Stone_scan_kernel_exclusive(const float *X, float *Y, const int len){
    __shared__ float XY[SECTION_SIZE+1];
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i == 0){
        XY[threadIdx.x] = 0.0f;
    } else if (i < len + 1){
        XY[threadIdx.x] = X[i - 1];
    } else{
        XY[threadIdx.x] = 0.0f;
    }
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2){
        __syncthreads();
        float tmp;
        if(threadIdx.x >= stride){
            tmp = XY[threadIdx.x] + XY[threadIdx.x - stride];
            XY[threadIdx.x] = tmp;
        }
        __syncthreads();
    }
    if (i < len + 1){
        Y[i] = XY[threadIdx.x];
    }
}


// __global__ void Kogge_Stone_hierarchical_kernel(const float *X, float *Y, const int len){
//     int i = blockDim.x * blockIdx.x + threadIdx.x;
//     if (i % HIERARCHICAL_SIZE == 0 && i < len){
//         Y[i / HIERARCHICAL_SIZE] = X[i];
//         for (unsigned int stride = 1; stride < HIERARCHICAL_SIZE; stride++){
//             __syncthreads();
//             if (i + stride < len){
//                 Y[i / HIERARCHICAL_SIZE] += X[i + stride];
//             }
//         }
//     }
// }

// __global__ void hierarchical_sum(const float *X, float *Y, const int len){
//     int i = blockDim.x * blockIdx.x + threadIdx.x;
//     if (i > HIERARCHICAL_SIZE){
//         Y[i] += X[i / HIERARCHICAL_SIZE];
//     }
// }

#define HIERARCHICAL_SIZE 2
__global__ void segemnt_sum(const float *X, float *Y, int len){
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float sData[HIERARCHICAL_SIZE];
    if (threadIdx.x < HIERARCHICAL_SIZE && i < len) {
        sData[threadIdx.x] = X[i];
    } else if (threadIdx.x < HIERARCHICAL_SIZE) {
        sData[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    if (threadIdx.x % HIERARCHICAL_SIZE != 0){
        sData[threadIdx.x] += sData[threadIdx.x - 1];
    }
    __syncthreads();
    if (i < len){
        Y[i] = sData[threadIdx.x];
    }
}

__global__ void featured_sum(const float *X, float *Y, int len) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    extern __shared__ float sData[];


}

void prefix_sum(const float *In, float *Out, float *Out_exclusive, float *Out_Inner, int len){
    float *d_In, *d_Out;
    int size_In = sizeof(float) * len;
    cudaMalloc((void**)&d_In, size_In);
    cudaMalloc((void**)&d_Out, size_In);
    cudaMemcpy(d_In, In, size_In, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (len + threadsPerBlock - 1) / threadsPerBlock;
    // Kogge_Stone_scan_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_In, d_Out, len);
    // cudaMemcpy(Out, d_Out, size_In, cudaMemcpyDeviceToHost);
    // coutarray(Out, len, "Prefix sum result");

    // cudaFree(d_Out);
    // cudaMalloc((void**)&d_Out, size_In);
    // Kogge_Stone_scan_kernel_double_buffer<<<blocksPerGrid, threadsPerBlock>>>(d_In, d_Out, len);
    // cudaMemcpy(Out, d_Out, size_In, cudaMemcpyDeviceToHost);
    // coutarray(Out, len, "Prefix sum double-buffer");

    // cudaFree(d_Out);
    // cudaMalloc((void**)&d_Out,  size_In + sizeof(float));
    // int exclusive_blocksPerGrid = (len + 1 + threadsPerBlock - 1) / threadsPerBlock;
    // Kogge_Stone_scan_kernel_exclusive<<<exclusive_blocksPerGrid, threadsPerBlock>>>(d_In, d_Out, len);
    // // std::cout << sizeof(d_Out) << std::endl;
    // cudaMemcpy(Out_exclusive, d_Out, size_In + sizeof(float), cudaMemcpyDeviceToHost);
    // coutarray(Out_exclusive, len+1, "Prefix sum in exclusive method");

    // cudaFree(d_Out);

    cudaFree(d_Out);
    int threadsHier = HIERARCHICAL_SIZE;
    int blocksHier = (len + threadsHier - 1) / threadsHier;
    cudaMalloc((void**)&d_Out, len * sizeof(float));
    segemnt_sum<<<blocksHier, threadsHier>>>(d_In, d_Out, len);
    cudaMemcpy(Out, d_Out, len * sizeof(float), cudaMemcpyDeviceToHost);
    coutarray(Out, len, "Step 1: sum the segement array");

    float *d_Inner;
    int size_inner = (len + HIERARCHICAL_SIZE - 1) / HIERARCHICAL_SIZE;
    cudaMalloc((void**)&d_Inner, size_inner * sizeof(float));
    int threadsFeature = 256;
    int blocksFeature = (size_inner + threadsFeature - 1) / threadsFeature;
    size_t sharedMemSize = size_inner * sizeof(float);
    std::cout << size_inner << std::endl;
    featured_sum<<<blocksFeature, threadsFeature, sharedMemSize>>>(d_Inner, d_Out, size_inner);
    cudaMemcpy(Out_Inner, d_Inner, size_inner * sizeof(float), cudaMemcpyDeviceToHost);
    coutarray(Out_Inner, size_inner, "Step 2: featrued data");


    // int threadsHier = 256;
    // int hierarchical_len = (len + HIERARCHICAL_SIZE - 1) / (HIERARCHICAL_SIZE);
    // int blocksHier = (hierarchical_len + threadsHier - 1) / threadsHier;
    // cudaMalloc((void**)&d_Out, hierarchical_len * sizeof(float));
    // Kogge_Stone_hierarchical_kernel<<<blocksHier, threadsHier>>>(d_In, d_Out, len);
    // // std::cout << (len + HIERARCHICAL_SIZE - 1) / (HIERARCHICAL_SIZE)  << std::endl;
    // cudaMemcpy(Out, d_Out, hierarchical_len * sizeof(float), cudaMemcpyDeviceToHost);
    // std::cout << "In segment of: " << HIERARCHICAL_SIZE << std::endl;
    // coutarray(Out, hierarchical_len, "First step: hierarchical scan sum");

    // // std::cout << hierarchical_len << std::endl;
    // float *new_Out;
    // cudaMalloc((void**)&new_Out, hierarchical_len * sizeof(float));
    // Kogge_Stone_scan_kernel<<<blocksHier, threadsHier>>>(d_Out, new_Out, hierarchical_len);
    // // std::cout << sizeof(d_Out) / sizeof(float) << std::endl;
    // cudaMemcpy(Out_Inner, new_Out, hierarchical_len * sizeof(float), cudaMemcpyDeviceToHost);
    // coutarray(Out_Inner, hierarchical_len, "Second step: prefix sum of hierarical scan");

    // hierarchical_sum<<<blocksPerGrid, threadsPerBlock>>>(new_Out, d_Out, len);
    // cudaMemcpy(Out, d_Out, size_In, cudaMemcpyDeviceToHost);
    // coutarray(Out, len, "Third step: final sum result");


    

    cudaFree(d_In);
    cudaFree(d_Out);
}

int main(){
    const float h_In[] = {1,2,3,4,5,6,7,8,9,0};
    // const float h_In[] = {1,2,3,1,-1,1,-1,1,-1,1};
    const int len = sizeof(h_In) / sizeof(float);
    float h_Out[len];
    float h_Out_exclusize[len+1];
    float h_Out_Inner[(len + HIERARCHICAL_SIZE - 1) / HIERARCHICAL_SIZE];

    coutarray(h_In, len, "Input");
    prefix_sum(h_In, h_Out, h_Out_exclusize, h_Out_Inner, len);

}