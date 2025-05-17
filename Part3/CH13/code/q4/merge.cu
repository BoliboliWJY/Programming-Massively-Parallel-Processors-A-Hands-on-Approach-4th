#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Define constants for block sizes
#define BLOCK_SIZE 256
#define MAX_CHUNK_SIZE 1024  // Max chunk size for local sorting

// Forward declarations of device functions
__device__ int countLE(int* arr, int start, int end, int val);
__device__ int findPosition(int* A, int startA, int endA, int* B, int startB, int endB, int target);

// Local sort kernel - sorts small chunks of data
__global__ void localSort(int* input, int* output, int n, int chunkSize) {
    extern __shared__ int sharedMem[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int chunkStart = blockIdx.x * chunkSize;
    
    // Load data into shared memory
    for (int i = tid; i < chunkSize && (chunkStart + i) < n; i += blockDim.x) {
        sharedMem[i] = input[chunkStart + i];
    }
    __syncthreads();
    
    // Sort data in shared memory using insertion sort (good for small chunks)
    for (int i = 1; i < chunkSize && (chunkStart + i) < n; i++) {
        int key = sharedMem[i];
        int j = i - 1;
        while (j >= 0 && sharedMem[j] > key) {
            sharedMem[j + 1] = sharedMem[j];
            j--;
        }
        sharedMem[j + 1] = key;
        __syncthreads(); // Ensure sorting step is complete before continuing
    }
    
    // Write sorted data back to global memory
    for (int i = tid; i < chunkSize && (chunkStart + i) < n; i += blockDim.x) {
        output[chunkStart + i] = sharedMem[i];
    }
}

// Parallel merge implementation from Chapter 12
__global__ void parallelMerge(int* A, int* B, int* C, int sizeA, int sizeB) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;
    int totalSize = sizeA + sizeB;
    int elemsPerThread = (totalSize + totalThreads - 1) / totalThreads;
    int start = min(gid * elemsPerThread, totalSize);
    int end = min(start + elemsPerThread, totalSize);
    
    if (start >= totalSize) return;
    
    // get start position
    int i = max(0, findPosition(A, 0, sizeA, B, 0, sizeB, start));
    int j = max(0, start - i); // ensure j is not negative
    
    // merge process
    int k = start;
    while (k < end) {
        if (j >= sizeB || (i < sizeA && A[i] <= B[j])) {
            C[k] = A[i];
            i++;
        } else {
            C[k] = B[j];
            j++;
        }
        k++;
    }
}

// Count elements less than or equal to given value
__device__ int countLE(int* arr, int start, int end, int val) {
    int left = start;
    int right = end;
    
    while (left < right) {
        int mid = (left + right) / 2;
        
        if (arr[mid] <= val) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    return left - start;
}

// Binary search to find position - core of parallel merge
__device__ int findPosition(int* A, int startA, int endA, int* B, int startB, int endB, int target) {
    // boundary check
    if (target <= 0) return startA;
    if (target >= (endA - startA) + (endB - startB)) return endA;
    
    int left = startA;
    int right = min(endA, startA + target + 1); // limit search range
    
    while (left < right) {
        int mid = left + (right - left) / 2;
        
        // A[mid]
        int aVal = A[mid];
        
        // count elements in B <= aVal
        int countB = 0;
        int bLeft = startB;
        int bRight = endB;
        while (bLeft < bRight) {
            int bMid = bLeft + (bRight - bLeft) / 2;
            if (B[bMid] <= aVal)
                bLeft = bMid + 1;
            else
                bRight = bMid;
        }
        countB = bLeft - startB;
        
        // count elements in A <= aVal
        int countA = mid - startA + 1;
        
        // total countA + countB elements <= aVal
        if (countA + countB <= target)
            left = mid + 1;
        else
            right = mid;
    }
    
    return left;
}

// Main sorting function
void mergeSortParallel(int* input, int* output, int n) {
    int* d_input;
    int* d_output;
    int* d_temp;
    
    // Allocate device memory
    cudaMalloc((void**)&d_input, n * sizeof(int));
    cudaMalloc((void**)&d_output, n * sizeof(int));
    cudaMalloc((void**)&d_temp, n * sizeof(int));
    
    // Copy input data to device
    cudaMemcpy(d_input, input, n * sizeof(int), cudaMemcpyHostToDevice);
    
    // Phase 1: Local sorting
    int chunkSize = MAX_CHUNK_SIZE;
    int numChunks = (n + chunkSize - 1) / chunkSize;
    int numThreadsPerChunk = min(chunkSize, BLOCK_SIZE);
    
    localSort<<<numChunks, numThreadsPerChunk, chunkSize * sizeof(int)>>>(
        d_input, d_output, n, chunkSize);
    
    // Phase 2: Recursively merge sorted chunks
    int* d_in = d_output;
    int* d_out = d_temp;
    
    for (int currSize = chunkSize; currSize < n; currSize *= 2) {
        int blocksNeeded = (n + 2 * currSize - 1) / (2 * currSize);
        
        for (int i = 0; i < blocksNeeded; i++) {
            int start = i * 2 * currSize;
            int mid = min(start + currSize, n);
            int end = min(start + 2 * currSize, n);
            
            // Use parallel merge from Chapter 12
            dim3 gridDim((end - start + BLOCK_SIZE - 1) / BLOCK_SIZE);
            dim3 blockDim(BLOCK_SIZE);
            
            parallelMerge<<<gridDim, blockDim>>>(
                d_in + start, d_in + mid, 
                d_out + start, mid - start, end - mid);
        }
        
        // Swap input and output arrays
        int* temp = d_in;
        d_in = d_out;
        d_out = temp;
    }
    
    // Copy results back to host
    cudaMemcpy(output, d_in, n * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp);
}

// Print a sample of the array contents
void printArraySample(int* arr, int size, const char* label) {
    printf("%s: \n", label);
    
    // Print first 10 elements
    printf("First 10 elements: ");
    int frontCount = min(10, size);
    for (int i = 0; i < frontCount; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
    
    // Print middle 10 elements (if size is sufficient)
    if (size > 20) {
        printf("Middle 10 elements: ");
        int midStart = size / 2 - 5;
        for (int i = 0; i < 10; i++) {
            printf("%d ", arr[midStart + i]);
        }
        printf("\n");
    }
    
    // Print last 10 elements (if size is sufficient)
    if (size > 10) {
        printf("Last 10 elements: ");
        int backCount = min(10, size);
        for (int i = size - backCount; i < size; i++) {
            printf("%d ", arr[i]);
        }
        printf("\n");
    }
    
    printf("\n");
}

// Main function
int main() {
    const int N = 1024 * 1024; // One million elements
    int* input = new int[N];
    int* output = new int[N];
    
    // Initialize random number generator
    srand(time(NULL));
    
    // Initialize array with random data
    for (int i = 0; i < N; i++) {
        input[i] = rand() % 10000;
    }
    
    // Display input array sample
    printf("======= BEFORE SORTING =======\n");
    printArraySample(input, N, "Input array");
    
    // Record sort start time
    clock_t start_time = clock();
    
    // Sort
    mergeSortParallel(input, output, N);
    
    // Record sort end time
    clock_t end_time = clock();
    double time_spent = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    
    // Display sorted result sample
    printf("======= AFTER SORTING =======\n");
    printArraySample(output, N, "Sorted array");
    
    // Validate sorting result
    bool sorted = true;
    for (int i = 1; i < N; i++) {
        if (output[i] < output[i-1]) {
            sorted = false;
            printf("Sort error: output[%d]=%d > output[%d]=%d\n", 
                   i-1, output[i-1], i, output[i]);
            break;
        }
    }
    
    if (sorted) {
        printf("Sort successful!\n");
    }
    
    // Display sorting time
    printf("Sorting time: %.4f seconds\n", time_spent);
    
    // Clean up
    delete[] input;
    delete[] output;
    
    return 0;
}