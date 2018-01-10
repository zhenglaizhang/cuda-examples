#include <iostream>
#include <math.h>


// Tells the CUDA C++ compiler that this is a function (kernel) that runs on the GPU and can be called from CPU code
// These __global__ functions are known as kernels, and code that runs on the GPU is often called device code, while code that runs on the CPU is host code.
__global__ void add(int n, float *x, float *y) {
    for (int i = 0; i < n; ++i) {
        y[i] = x[i] + y[i];
    }
}

int main() {
    int N = 1 << 20; // 1M
    float *x, *y;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; ++i) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }


    // Run kernel on 1M elements on the GPU
    add<<<1, 1>>>(N, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    float max_error = 0.0f;
    for (int i = 0; i < N; ++i) {
        max_error = fmax(max_error, fabs(y[i] - 3.0f));
    }
    std::cout << "max error: " << max_error << std::endl;

    // free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}