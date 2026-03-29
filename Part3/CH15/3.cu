#include <iostream>
#include <vector>
#include <string>
#include <cstdio>
#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)           \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)


#include <limits>
constexpr int INF = std::numeric_limits<int>::max();

struct CsrGraph{
    int numVertices;
    int numEdges;
    std::vector<int> srcPtrs;
    std::vector<int> dst;
};

std::vector<int> loadGraph(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("open failed: " + path);
    std::fseek(f, 0, SEEK_END);
    long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (bytes % sizeof(int) != 0) throw std::runtime_error("file size mismatch");
    std::vector<int> data(bytes / sizeof(int));
    size_t read = std::fread(data.data(), sizeof(int), data.size(), f);
    std::fclose(f);
    if (read != data.size()){
        throw std::runtime_error("short read: " + path);
    }
    return data;
}

__global__ void regularFrontierBfsKernel(
    int* d_srcPtrs,
    int* d_dst,
    int* d_currFrontier,
    int currSize,
    int currLevel,
    int* d_level,
    int* d_nextFrontier,
    int* d_nextSize
        ){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < currSize){
        int u = d_currFrontier[i];
        for (int j = d_srcPtrs[u]; j < d_srcPtrs[u + 1]; j++) {

            int v = d_dst[j];
            if(atomicCAS(&d_level[v], INF, currLevel + 1) == INF){
                int pos = atomicAdd(d_nextSize, 1);
                d_nextFrontier[pos] = v;
            }
        }
    }
}

__global__ void single_blockKernel(int* d_srcPtrs, int* d_dst, int* d_currFrontier, int currSize, int currLevel, int* d_level, int* d_nextFrontier, int* d_nextSize, int* d_overflow, int* d_levelsProcessed){
    constexpr int CAP = 256;

    __shared__ int s_currFrontier[CAP];
    __shared__ int s_nextFrontier[CAP];
    __shared__ int s_currSize;
    __shared__ int s_nextSize;
    __shared__ int s_overflow;
    __shared__ int s_currLevel;
    __shared__ int s_levelsProcessed;
    __shared__ int s_exit;

    int tid = threadIdx.x; 
    // thread0 initializes shared variables
    if (tid == 0) { 
        s_currSize = currSize;
        s_currLevel = currLevel;
        s_nextSize = 0;
        s_overflow = 0;
        s_levelsProcessed = 0;
        s_exit = 0;
    }

    if (tid < currSize) {
        s_currFrontier[tid] = d_currFrontier[tid];
    }
    __syncthreads();

    while (true) {
        if(tid == 0) {
            s_nextSize = 0;
            s_overflow = 0;
            s_exit = 0;
        }
        __syncthreads();
        
        if (tid < s_currSize) {
            int u = s_currFrontier[tid]; // current top
            for (int j = d_srcPtrs[u]; j < d_srcPtrs[u + 1]; j++) {
                int v = d_dst[j]; // scan u's neighbors
                if (atomicCAS(&d_level[v], INF, s_currLevel + 1) == INF) {
                    int pos = atomicAdd(&s_nextSize, 1);
                    d_nextFrontier[pos] = v;
                    if (pos < CAP) {
                        s_nextFrontier[pos] = v;
                    } else {
                        atomicExch(&s_overflow, 1);
                    }
                }
            }
        }
        __syncthreads();

        if (tid == 0) {
            s_levelsProcessed++;
            *d_nextSize = s_nextSize;
            *d_overflow = s_overflow;
            *d_levelsProcessed = s_levelsProcessed;
    
            if (s_nextSize == 0 || s_overflow) {
                s_exit = 1;
            } else {
                s_currSize = s_nextSize;
                s_currLevel++;
            }
        }
        __syncthreads();

        if (s_exit) {
            return;
        }

        if (tid < s_currSize) {
            s_currFrontier[tid] = s_nextFrontier[tid];
        }
        __syncthreads();
    }
}

int main(){
    CsrGraph g;
    const std::string dataDir = "data";
    g.srcPtrs = loadGraph(dataDir + "/srcPtrs.bin");
    g.dst = loadGraph(dataDir + "/dst.bin");
    g.numVertices = static_cast<int>(g.srcPtrs.size()) - 1;
    g.numEdges = static_cast<int>(g.dst.size());
    std::cout << "Graph loaded: " << g.numVertices << " vertices, " << g.numEdges << " edges" << std::endl;

    int *d_srcPtrs = nullptr, *d_dst = nullptr, *d_level = nullptr, *d_currFrontier = nullptr, *d_nextFrontier = nullptr, *d_nextSize = nullptr, *d_overflow = nullptr, *d_levelsProcessed = nullptr;
    CUDA_CHECK(cudaMalloc(&d_srcPtrs, g.srcPtrs.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_srcPtrs, g.srcPtrs.data(), g.srcPtrs.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_dst, g.dst.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_dst, g.dst.data(), g.dst.size() * sizeof(int), cudaMemcpyHostToDevice));
    
    int source = 0;
    int numVertices = g.numVertices;
    std::vector<int> h_level(numVertices, INF);
    h_level[source] = 0;
    CUDA_CHECK(cudaMalloc(&d_level, h_level.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_level, h_level.data(), h_level.size() * sizeof(int), cudaMemcpyHostToDevice));

    
    int currSize = 1;
    int currLevel = 0;
    int nextSize = 0;

    //最坏一层会有numVetrices个顶点

    CUDA_CHECK(cudaMalloc(&d_currFrontier, numVertices * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nextFrontier, numVertices * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nextSize, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_currFrontier, &source, sizeof(int), cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (currSize + threadsPerBlock - 1) / threadsPerBlock;
    
    int h_overflow, h_levelsProcessed;
    CUDA_CHECK(cudaMalloc(&d_overflow, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_levelsProcessed, sizeof(int)));


    // 启动
    while (currSize > 0){
        CUDA_CHECK(cudaMemset(d_nextSize, 0, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_overflow, 0, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_levelsProcessed, 0, sizeof(int)));
        bool use_singleBlock = false;
        if (currSize <= threadsPerBlock) {
            single_blockKernel<<<1, threadsPerBlock>>>(
                d_srcPtrs,
                d_dst,
                d_currFrontier,
                currSize,
                currLevel,
                d_level,
                d_nextFrontier,
                d_nextSize,
                d_overflow,
                d_levelsProcessed
            );
            use_singleBlock = true;
        } else {
            regularFrontierBfsKernel<<<blocksPerGrid, threadsPerBlock>>>(
                d_srcPtrs,
                d_dst,
                d_currFrontier,
                currSize,
                currLevel,
                d_level,
                d_nextFrontier,
                d_nextSize
            );
            use_singleBlock = false;
        }
        
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&nextSize, d_nextSize, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_levelsProcessed, d_levelsProcessed, sizeof(int), cudaMemcpyDeviceToHost));

        std::cout << "currLevel=" << currLevel
        << ", currSize=" << currSize
        << ", nextSize=" << nextSize
        << ", use_singleBlock=" << use_singleBlock
        << ", levelsProcessed=" << h_levelsProcessed
        << ", overflow=" << h_overflow
        << std::endl;

        if (nextSize == 0){
            break;
        }
        std::swap(d_currFrontier, d_nextFrontier);
        currSize = nextSize;
        if (use_singleBlock){
            currLevel += h_levelsProcessed;
        } else {
            currLevel += 1;
        }
        blocksPerGrid = (currSize + threadsPerBlock - 1) / threadsPerBlock;

    }

    CUDA_CHECK(cudaMemcpy(h_level.data(), d_level, numVertices * sizeof(int), cudaMemcpyDeviceToHost));

    int visited = 0;
    int maxLevel = 0;
    for (int lv : h_level) {
        if (lv != INF) {
            visited++;
            maxLevel = std::max(maxLevel, lv);
        }
    }
    std::cout << "Visited vertices: " << visited << std::endl;
    std::cout << "Max levels: " << maxLevel << std::endl;
    cudaFree(d_level);
    cudaFree(d_srcPtrs);
    cudaFree(d_dst);
    cudaFree(d_currFrontier);
    cudaFree(d_nextFrontier);
    cudaFree(d_nextSize);
    cudaFree(d_overflow);
    cudaFree(d_levelsProcessed);
    return 0;
    

}