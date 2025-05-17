// q1
__global__ void radix_sort_iter_memory_coalescing(unsigned int* input, unsigned int* output, unsigned int* bits, unsigned int N, unsigned int iter) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;

    // shared memory
    extern __shared__ unsigned int s_mem[];
    unsigned int* s_data = s_mem;
    unsigned int* s_bits = &s_data[blockDim.x];

    unsigned int key = 0, bit = 0;
    if(i < N) {
        key = input[i];
        s_data[tid] = key;
        bit = (key >> iter) & 1;
        s_bits[tid] = bit;
        bits[i] = bit;
    }
    __syncthreads();

    exclusiveScan(bits, N);

    if(i < N) {
        unsigned int numOnesBefore = bits[i], numOnesTotal = bits[N];
        unsigned int dst;
        if (bit == 0) {
            dst = i - numOnesBefore;
        } else {
            dst = N - numOnesTotal - (i - numOnesBefore);
        }
        output[dst] = s_data[tid]; // load data from shared memory
    }
}

// q2
__global__ void radix_sort_iter_multibit(unsigned int* input, unsigned int* output, unsigned int*bits, unsigned int N, unsigned int iter, unsigned int numBits) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;
    
    // shared memory
    extern __shared__ unsigned int s_mem[];
    unsigned int* s_data = s_mem;
    unsigned int* s_digits = &s_data[blockDim.x];

    // mask
    unsigned int mask = (1 << numBits) - 1; // when numBits = 2, mask = 0b11
    unsigned int digit = 0;
    unsigned int key = 0;

    if(i < N) {
        key = input[i];
        s_data[tid] = key;
        digit = (key >> iter) & mask; // get more digits
        s_digits[tid] = digit;
    }
    __syncthreads();
    
    if(i < N) {
        unsigned int dst = atomicAdd(&counters[digit], 1);
        output[dst] = s_data[tid];
    }
}

// q3
__global__ void radix_sort_iter_thread_coarsening(unsigned int* input, unsigned int* output, unsigned int* bits, unsigned int N, unsigned int iter, unsigned int numBits, unsigned int elements_per_thread) {
    unsigned int i_base = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;

    // shared memory
    extern __shared__ unsigned int s_mem[];
    unsigned int* s_data = s_mem;
    unsigned int* s_digits = &s_data[blockDim.x * elements_per_thread]; // menory size should be elements_per_thread larger

    // mask
    unsigned int mask = (1 << numBits) - 1;
    
    #pragma unroll
    for (int e = 0; e < elements_per_thread; e++) {
        unsigned int i = i_base + e * blockDim.x;
        
        if (i < N) {
            unsigned int key = input[i];
            s_data[tid + e * blockDim.x] = key;
            
            unsigned int digit = (key >> iter) & mask;
            s_digits[tid + e * blockDim.x] = digit;
            
            bits[i] = digit;
        }
    }
    __syncthreads();

    exclusiveScan(bits, N);

    #pragma unroll
    for (int e = 0; e < elements_per_thread; e++) {
        unsigned int i = i_base + e * blockDim.x;

        if (i < N) {
            unsigned int digit = s_digits[tid + e * blockDim.x];
            unsigned int numOnesBefore = bits[i], numOnesTotal = bits[N];
            unsigned int dst;
            if (digit == 0) {
                dst = i - numOnesBefore;
            } else {
                dst = N - numOnesTotal - (i - numOnesBefore);
            }
            output[dst] = s_data[tid + e * blockDim.x];
        }
    }
}
