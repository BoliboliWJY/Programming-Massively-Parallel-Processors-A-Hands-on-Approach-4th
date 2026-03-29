#include <iostream>
#include <vector>
#include <string>
#include <cstdio>
#include <stdexcept>
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
/*
newVertexVisited用于检查这一轮是否还有新发现
有线程标记了，则记为1，继续；否则没有新发现，结束while
初始化，level都设定为UINT_MAX，表示未访问过，只有起始设定为0
if (level[neighbour] == INT_MAX)用于判断是否访问过

切换方向 pull or push?
根据frontier大小判断
当前层活跃节点数 / 图的总顶点数 > α， 切换为pull
*/

struct CsrGraph{
    int numVertices;
    int numEdges;
    std::vector<int> srcPtrs;
    std::vector<int> dst;
    std::vector<int> dstPtrs;
    std::vector<int> src;
};

__global__ void bfs_push_kernel(int* srcPtrs, int* dst, int* level, int* d_count, int currLevel, int numVertices){
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u < numVertices){
        if (level[u] == currLevel - 1){//只有上一层才推
            for(int i = srcPtrs[u]; i < srcPtrs[u + 1]; i++){
                int v = dst[i];
                int old = atomicCAS(&level[v], INF, currLevel); //多个线程可能同时看到 level[v] == INF，然后都去执行 CAS
                if (old == INF){
                    atomicAdd(d_count, 1);//只标记一次即可 
                }
            }
        }
    }
}

__global__ void bfs_pull_kernel(int* dstPtrs, int* src, int* level, int* d_count, int currLevel, int numVertices){
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if(v < numVertices){
        if (level[v] == INF){
            for (int i = dstPtrs[v]; i < dstPtrs[v + 1]; i++){
                int u = src[i];//往上找
                if (level[u] == currLevel - 1) {
                    level[v] = currLevel;
                    atomicAdd(d_count, 1);
                    break;
                }
            }
        }
    }
}

std::vector<int> readBinaryInts(const std::string& path) {
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

int main(){
    // 读取数据

    CsrGraph g;
    const std::string dataDir = "data";
    g.srcPtrs = readBinaryInts(dataDir + "/srcPtrs.bin");
    g.dst = readBinaryInts(dataDir + "/dst.bin");
    g.dstPtrs = readBinaryInts(dataDir + "/dstPtrs.bin");
    g.src = readBinaryInts(dataDir + "/src.bin");
    g.numVertices = static_cast<int>(g.srcPtrs.size()) - 1;
    g.numEdges = static_cast<int>(g.dst.size());
    std::cout << "Graph loaded: " << g.numVertices << " vertices, " << g.numEdges << " edges" << std::endl;



    int *d_srcPtrs = nullptr, *d_dst = nullptr, *d_dstPtrs = nullptr, *d_src = nullptr;
    CUDA_CHECK(cudaMalloc(&d_srcPtrs, g.srcPtrs.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_srcPtrs, g.srcPtrs.data(), g.srcPtrs.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_dst, g.dst.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_dst, g.dst.data(), g.dst.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_dstPtrs, g.dstPtrs.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_dstPtrs, g.dstPtrs.data(), g.dstPtrs.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_src, g.src.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_src, g.src.data(), g.src.size() * sizeof(int), cudaMemcpyHostToDevice));
    
    int numVertices = g.numVertices;
    float alpha = 0.1;
    int threshold = static_cast<int>(numVertices * alpha);
    std::vector<int> h_level(numVertices, INF);
    int source = 0;
    h_level[source] = 0;
    int *d_level = nullptr, *d_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_level, numVertices * sizeof(int))); //直接用nunVertices，避免直接用h_level.size()出现越界的问题
    CUDA_CHECK(cudaMemcpy(d_level, h_level.data(), numVertices * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));


    int h_count = 1;
    int currLevel = 1;
    int maxLevel = 0;
    int threadsPerBlock = 256;
    int blocksPerGrid = (numVertices + threadsPerBlock - 1) / threadsPerBlock;

    while (h_count > 0){
        CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

        if (h_count > threshold){ //优先pull
            std::cout << "Level " << currLevel << ": Using PULL mode (h_count=" << h_count << ")" << std::endl;
            bfs_pull_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_dstPtrs, d_src, d_level, d_count, currLevel, g.numVertices);
        } else {
            std::cout << "Level " << currLevel << ": Using PUSH mode (h_count=" << h_count << ")" << std::endl;
            bfs_push_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_srcPtrs, d_dst, d_level, d_count, currLevel, g.numVertices);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));

        if (h_count > 0) {
            maxLevel = currLevel;
        }

        currLevel++;
        if (currLevel > numVertices) break;
    }
    std::cout << "BFS max level = " << maxLevel << std::endl;

    CUDA_CHECK(cudaMemcpy(h_level.data(), d_level, numVertices * sizeof(int), cudaMemcpyDeviceToHost));

    int visited = 0;
    for (int lv : h_level) {
        if (lv != INF) {
            visited++;
        }
    }
    std::cout << "Visited vertices: " << visited << std::endl;

    cudaFree(d_level);
    cudaFree(d_count);
    cudaFree(d_srcPtrs);
    cudaFree(d_dst);
    cudaFree(d_dstPtrs);
    cudaFree(d_src);
    return 0;
}