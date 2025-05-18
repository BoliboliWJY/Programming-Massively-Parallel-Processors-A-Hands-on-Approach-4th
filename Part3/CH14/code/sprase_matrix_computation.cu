// q3
__global__ void computeHistogram(int* rowIdx, int* rowNnz, int nnz) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < nnz) {
        int row = rowIdx[tid];
        atomicAdd(&rowNnz[row], 1);
    }
}

__global__ void exclusiveScan(int* rowNnz, int* rowPtrs, int numRows) {
    extern __shared__ int temp[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < numRows) {
        temp[tid] = rowNnz[idx];
    } else {
        temp[tid] = 0;
    }
    __syncthreads();

    // prefix sum
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            temp[index] += temp[index - stride];
        }
        __syncthreads();
    }

    if (idx < numRows) {
        if (tid > 0) {
            rowPtrs[idx] = temp[tid - 1];
        } else {
            rowPtrs[idx] = 0; // first row pointer is 0
        }
    }
}

__global__ void reorderElements(int* cooRowIdx, int* cooColIdx, float* cooValues, int* csrRowPtrs, int* csrColIdx, float* csrValues, int* rowOffsets, int nnz) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < nnz) {
        int row = cooRowIdx[tid];
        int offset = atomicAdd(&rowOffsets[row], 1);
        int pos = csrRowPtrs[row] + offset;

        csrColIdx[pos] = cooColIdx[tid];
        csrValues[pos] = cooValues[tid];
    }
}

// q4
__global__ void spmvELL(int numRows, int maxColsPerRow, int* colIdx, float* values, float* x, float* y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows) {
        float sum = 0.0f;

        for (int i = 0; i < maxColsPerRow; i++) {
            int idx = i * numRows + row;
            int col = colIdx[idx];

            // ignore elements filled with -1
            if (col >= 0) {
                sum += values[idx] * x[col];
            }
        }

        y[row] = sum;
    }
}

// q5
__global__ void spmvJDS(int numRows, int *rowPerm, int *jdsRowPtrs, int numJdsDiagonals, int *colIdx, float *values, float *x, float *y) {
    tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < numRows) {
        float sum = 0.0f;

        for (int j = 0; j < numJdsDiagonals; j++) {
            if (j < numJdsDiagonals && tid < jdsRowPtrs[j+1] - jdsRowPtrs[j]) {
                int idx = jdsRowPtrs[j] + tid;
                int col = colIdx[idx];

                if (col >= 0) {
                    sum += values[idx] * x[col];
                }
            }
        }

        int originalRow = rowPerm[row];
        y[originalRow] = sum;
    }
}