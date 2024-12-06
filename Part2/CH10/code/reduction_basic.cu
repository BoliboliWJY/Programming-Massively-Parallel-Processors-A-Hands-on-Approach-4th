#include <iostream>
#include <float.h>
void coutarray(const float* arr, int len, const char* name){
    std::cout << "Array " << name << ":\n";
    for (int i = 0; i < len; i++){
        std::cout << arr[i] << "\t";
    }
    std::cout << "\n";
}


__global__ void reduction_squencial(float *In, float *Out, const int len){
    __shared__ float sdata[256];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = (idx < len) ? In[idx] : 0;
    sdata[threadIdx.x] = val;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if (threadIdx.x < stride){
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0){
        Out[blockIdx.x] = sdata[0];
    }
}

__global__ void reduction_reverse(float *In, float *Out, const int len){
    __shared__ float sdata[256];
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < len) ? In[idx] : 0;
    sdata[threadIdx.x] = val;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if (threadIdx.x >= stride){
            sdata[threadIdx.x] += sdata[threadIdx.x - stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == (blockDim.x - 1)){
        Out[blockIdx.x] = sdata[threadIdx.x];
    }
}

#define COARSE_FACTOR 3
__global__ void reduction_coarsened_sum(float *In, float *Out, int len){
    const int BLOCKDIM = 256;
    __shared__ float input_s[BLOCKDIM];
    unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;
    float sum = 0.0f;
    if (i < len) sum += In[i];
    for (unsigned int tile = 1; tile < COARSE_FACTOR * 2; tile++){
        unsigned int idx = i + tile * BLOCKDIM;
        if (idx < len)  sum += In[i + tile * BLOCKDIM];
        
    }
    input_s[t] = sum;
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride >>= 1){
        __syncthreads();
        if (t < stride){
            input_s[t] += input_s[t + stride];
        }
    }
    __syncthreads();
    if (t == 0){
        atomicAdd(Out, input_s[0]);
    }
}

__device__ float atomicMaxFloat(float *address, float val) {
    //used to compare with float type data and return the max data
    int *address_as_int = (int*) address;
    int old = *address_as_int, assumed;

    do {
        assumed = old;
        float old_val = __int_as_float(assumed);
        float max_val = fmaxf(old_val, val);
        int new_val_int = __float_as_int(max_val);
        old = atomicCAS(address_as_int, assumed, new_val_int);
    } while (assumed != old);

    return __int_as_float(old);
}

__global__ void reduction_coarsened_max(float *In, float *Out, int len){
    const int BLOCKDIM = 256;
    __shared__ float input_s[BLOCKDIM];
    unsigned int segment = COARSE_FACTOR * 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;
    unsigned int t = threadIdx.x;
    float max_num = -FLT_MAX;
    if (i < len){
        max_num = In[i];
    }
    for (unsigned int tile = 1; tile < COARSE_FACTOR * 2; tile++){
        unsigned int idx = i + tile * BLOCKDIM;
        if (idx < len)  if(max_num < In[i + tile * BLOCKDIM])   max_num = In[i + tile * BLOCKDIM]; 
        
    }
    input_s[t] = max_num;
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride >>= 1){
        __syncthreads();
        if (t < stride){
            if (input_s[t] < input_s[t + stride])    input_s[t] = input_s[t + stride];
        }
    }
    __syncthreads();
    if (t == 0){
        atomicMaxFloat(Out, input_s[0]);
    }
}

void reduction(const float *In, float *Out, const int len){
    float *d_In, *d_Out;
    cudaMalloc((void**)&d_In, len * sizeof(float));

    cudaMemcpy(d_In, In, len * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (len + threadsPerBlock - 1) / threadsPerBlock;
    cudaMalloc((void**)&d_Out, blocksPerGrid * sizeof(float));

    reduction_squencial<<<blocksPerGrid, threadsPerBlock>>>(d_In, d_Out, len);
    int size = blocksPerGrid;
    while (size > 1){
        int newBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
        reduction_squencial<<<newBlocks, threadsPerBlock>>>(d_Out, d_Out, size);
        size = newBlocks;
    }
    // std::cout << len << std::endl;
    cudaMemcpy(Out, d_Out, sizeof(float), cudaMemcpyDeviceToHost);
    coutarray(Out, 1, "Output_squencial");

    cudaFree(d_Out);
    reduction_reverse<<<blocksPerGrid, threadsPerBlock>>>(d_In, d_Out, len);
    cudaMemcpy(Out, d_Out, sizeof(float), cudaMemcpyDeviceToHost);
    coutarray(Out, 1, "Output_reverse");
    size = blocksPerGrid;
    while (size > 1){
        int newBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
        reduction_reverse<<<newBlocks, threadsPerBlock>>>(d_Out, d_Out, size);
        size = newBlocks;
    }
    
    cudaFree(d_Out);
    reduction_coarsened_sum<<<blocksPerGrid, threadsPerBlock>>>(d_In, d_Out, len);
    cudaMemcpy(Out, d_Out, sizeof(float), cudaMemcpyDeviceToHost);
    coutarray(Out, 1, "Output_tiled_sum");

    cudaFree(d_Out);
    float initVal = -FLT_MAX;
    cudaMalloc((void**)&d_Out, sizeof(float));//reallocate the data
    cudaMemcpy(d_Out, &initVal, sizeof(float), cudaMemcpyHostToDevice); //initialize the regional number
    reduction_coarsened_max<<<blocksPerGrid, threadsPerBlock>>>(d_In, d_Out, len);
    cudaMemcpy(Out, d_Out, sizeof(float), cudaMemcpyDeviceToHost);
    coutarray(Out, 1, "Output_tiled_max");

    std::cout<< "For a len that is not the multiple of coarse factor, just use set the exceeded number as 0." << std::endl;

    cudaFree(d_In);
    cudaFree(d_Out);

}

int main(){
    const float h_In[] = {1,2,3,4,5,6,7,8,9,0,1,2,3,4,5};
    const int len = sizeof(h_In) / sizeof(float);
    // const int size_In = len;
    // std::cout << size_In << "\n";
    float h_Out;

    coutarray(h_In, len, "Input");
    reduction(h_In, &h_Out, len);

    return 0;
}