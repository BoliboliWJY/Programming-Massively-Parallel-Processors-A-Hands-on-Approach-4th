#include "CudaMatrix.h"

namespace CudaMatrix {

// Print matrix function implementation
void coutMatrix(const float* mat, int rows, int cols, const char* name) {
    std::cout << "Matrix " << name << ":\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << mat[i * cols + j] << "\t";
        }
        std::cout << "\n";
    }
}

} // namespace CudaMatrix
