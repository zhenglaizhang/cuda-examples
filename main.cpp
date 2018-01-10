#include <iostream>
#include <math.h>


void add(int n, float *x, float *y) {
    for (int i = 0; i < n; ++i) {
        y[i] = x[i] + y[i];
    }
}

int main() {
    int N = 1 << 20; // 1M
    float *x = new float[N];
    float *y = new float[N];
    for (int i = 0; i < N; ++i) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // run kernel on CPU
    add(N, x, y);

    float max_error = 0.0f;
    for (int i = 0; i < N; ++i) {
        max_error = fmax(max_error, fabs(y[i] - 3.0f));
    }
    std::cout << "max error: " << max_error << std::endl;

    delete[] x;
    delete[] y;

    return 0;
}