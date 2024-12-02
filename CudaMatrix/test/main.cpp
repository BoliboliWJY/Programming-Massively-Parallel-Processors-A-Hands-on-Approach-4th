#include "C:\Users\ASUS\Desktop\git\Programming Massively Parallel Processors A Hands-on Approach 4th\CudaMatrix\include\CudaMatrix.h"

int main() {
    const int rows = 2, cols = 3;
    float mat[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    // Call the coutMatrix function
    CudaMatrix::coutMatrix(mat, rows, cols, "TestMatrix");

    return 0;
}
