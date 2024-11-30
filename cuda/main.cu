#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// Define grid dimensions and coarsening factor
#define NX 128
#define NY 128
#define NZ 128
#define COARSENING_FACTOR_Z 4

// CUDA error checking macro
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
    if (code != cudaSuccess){
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// CUDA Kernel
__global__ void sevenPointStencilCoarsenedShared(
    const float* __restrict__ inPrev,
    const float* __restrict__ inCurr,
    const float* __restrict__ inNext,
    float* __restrict__ out,
    int nx, int ny, int nz)
{
    // Define shared memory dimensions (including halo)
    extern __shared__ float sharedMem[];
    float* sPrev = sharedMem;
    float* sCurr = &sharedMem[(blockDim.x + 2) * (blockDim.y + 2)];

    // Calculate global thread coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // Assuming blockDim.z = 1
    int z_start = blockIdx.z * COARSENING_FACTOR_Z;

    // Load data into shared memory with halo regions
    if (x < nx && y < ny && z_start < nz) {
        // Load inPrev and inCurr into shared memory
        sPrev[(threadIdx.y + 1) * (blockDim.x + 2) + (threadIdx.x + 1)] = inPrev[z_start * nx * ny + y * nx + x];
        sCurr[(threadIdx.y + 1) * (blockDim.x + 2) + (threadIdx.x + 1)] = inCurr[z_start * nx * ny + y * nx + x];
        
        // Load halo cells
        if (threadIdx.x == 0 && x > 0) {
            sPrev[(threadIdx.y + 1) * (blockDim.x + 2) + 0] = inPrev[z_start * nx * ny + y * nx + (x - 1)];
            sCurr[(threadIdx.y + 1) * (blockDim.x + 2) + 0] = inCurr[z_start * nx * ny + y * nx + (x - 1)];
        }
        if (threadIdx.x == blockDim.x - 1 && x < nx -1) {
            sPrev[(threadIdx.y + 1) * (blockDim.x + 2) + (blockDim.x +1)] = inPrev[z_start * nx * ny + y * nx + (x +1)];
            sCurr[(threadIdx.y + 1) * (blockDim.x + 2) + (blockDim.x +1)] = inCurr[z_start * nx * ny + y * nx + (x +1)];
        }
        if (threadIdx.y == 0 && y > 0) {
            sPrev[0 * (blockDim.x +2) + (threadIdx.x +1)] = inPrev[z_start * nx * ny + (y -1) * nx + x];
            sCurr[0 * (blockDim.x +2) + (threadIdx.x +1)] = inCurr[z_start * nx * ny + (y -1) * nx + x];
        }
        if (threadIdx.y == blockDim.y -1 && y < ny -1) {
            sPrev[(blockDim.y +1) * (blockDim.x +2) + (threadIdx.x +1)] = inPrev[z_start * nx * ny + (y +1) * nx + x];
            sCurr[(blockDim.y +1) * (blockDim.x +2) + (threadIdx.x +1)] = inCurr[z_start * nx * ny + (y +1) * nx + x];
        }
    }

    // Synchronize to ensure all shared memory loads are complete
    __syncthreads();

    // Initialize 'above' for the first Z layer
    float above = inNext[z_start * nx * ny + y * nx + x];

    // Loop over coarsened Z layers
    for (int cz = 0; cz < COARSENING_FACTOR_Z; ++cz) {
        int z = z_start + cz;

        // Boundary check
        if (x > 0 && x < nx - 1 && y > 0 && y < ny - 1 && z > 0 && z < nz - 1) {
            // Compute 1D index
            int idx = z * nx * ny + y * nx + x;

            // Load data from shared memory
            float center = sCurr[(threadIdx.y +1) * (blockDim.x +2) + (threadIdx.x +1)];
            float left   = sCurr[(threadIdx.y +1) * (blockDim.x +2) + threadIdx.x];
            float right  = sCurr[(threadIdx.y +1) * (blockDim.x +2) + (threadIdx.x +2)];
            float front  = sCurr[threadIdx.y * (blockDim.x +2) + (threadIdx.x +1)];
            float back   = sCurr[(threadIdx.y +2) * (blockDim.x +2) + (threadIdx.x +1)];
            float below  = sPrev[(threadIdx.y +1) * (blockDim.x +2) + (threadIdx.x +1)];
            // 'above' now references the correct neighbor
            // For cz = 0, it's from inNext; for cz > 0, it's from the previous 'out' computation
            // For cz > 0, 'above' should be the value computed in the previous iteration
            // Since 'out' is already written, we can read it back
            above = (cz == 0) ? inNext[idx] : out[idx - nx * ny];

            // Stencil computation
            out[idx] = (left + right + front + back + below + above) / 6.0f;
        }
    }

    // No need to synchronize here as threads are independent
}


int main(){
    // Define grid dimensions
    const int nx = NX;
    const int ny = NY;
    const int nz = NZ;

    // Total number of elements
    size_t totalSize = nx * ny * nz;

    // Allocate host memory
    std::vector<float> h_inPrev(totalSize, 1.0f); // Initialize with 1.0f for testing
    std::vector<float> h_inCurr(totalSize, 2.0f); // Initialize with 2.0f for testing
    std::vector<float> h_inNext(totalSize, 3.0f); // Initialize with 3.0f for testing
    std::vector<float> h_out(totalSize, 0.0f);    // Initialize output with 0.0f

    // Allocate device memory
    float *d_inPrev, *d_inCurr, *d_inNext, *d_out;
    cudaCheckError(cudaMalloc((void**)&d_inPrev, totalSize * sizeof(float)));
    cudaCheckError(cudaMalloc((void**)&d_inCurr, totalSize * sizeof(float)));
    cudaCheckError(cudaMalloc((void**)&d_inNext, totalSize * sizeof(float)));
    cudaCheckError(cudaMalloc((void**)&d_out,    totalSize * sizeof(float)));

    // Copy data from host to device
    cudaCheckError(cudaMemcpy(d_inPrev, h_inPrev.data(), totalSize * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_inCurr, h_inCurr.data(), totalSize * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_inNext, h_inNext.data(), totalSize * sizeof(float), cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    dim3 blockDim(8, 8, 1); // X, Y, Z dimensions
    dim3 gridDim(
        (nx + blockDim.x -1) / blockDim.x,
        (ny + blockDim.y -1) / blockDim.y,
        (nz + (blockDim.z * COARSENING_FACTOR_Z) -1) / (blockDim.z * COARSENING_FACTOR_Z)
    );

    // Calculate shared memory size
    size_t sharedMemSize = 2 * (blockDim.x + 2) * (blockDim.y + 2) * sizeof(float);

    // Launch the kernel
    sevenPointStencilCoarsenedShared<<<gridDim, blockDim, sharedMemSize>>>(
        d_inPrev, d_inCurr, d_inNext, d_out, nx, ny, nz
    );

    // Check for kernel launch errors
    cudaCheckError(cudaGetLastError());

    // Synchronize to ensure kernel completion
    cudaCheckError(cudaDeviceSynchronize());

    // Copy result back to host
    cudaCheckError(cudaMemcpy(h_out.data(), d_out, totalSize * sizeof(float), cudaMemcpyDeviceToHost));

    // Simple verification (for demonstration purposes)
    // Here, we expect out[idx] = (left + right + front + back + below + above) /6
    // Given initialization:
    // left, right, front, back, below = 2.0f (from inCurr)
    // above = 3.0f (from inNext)
    // So, out[idx] = (2 + 2 + 2 + 2 + 2 + 3) /6 = 13 /6 ≈ 2.1667
    bool correct = true;
    float expected = 13.0f / 6.0f;
    for(int z=1; z < nz-1 && correct; ++z){
        for(int y=1; y < ny-1 && correct; ++y){
            for(int x=1; x < nx-1 && correct; ++x){
                int idx = z * nx * ny + y * nx + x;
                if(abs(h_out[idx] - expected) > 1e-4){
                    std::cout << "Mismatch at (" << x << "," << y << "," << z << "): " 
                              << h_out[idx] << " != " << expected << std::endl;
                    correct = false;
                }
            }
        }
    }

    if(correct){
        std::cout << "Stencil computation successful. All values match expected results." << std::endl;
    } else {
        std::cout << "Stencil computation failed. Mismatched values found." << std::endl;
    }

    // Free device memory
    cudaFree(d_inPrev);
    cudaFree(d_inCurr);
    cudaFree(d_inNext);
    cudaFree(d_out);

    return 0;
}
