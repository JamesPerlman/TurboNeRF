#include <iostream>
#include <cuda_runtime.h>

#include <tiny-cuda-nn/common.h>

#include "main.h"

// Declare a global device function
__global__ void helloCuda(void)
{
    printf("Hello World from GPU!!\n");
}

int main()
{
    // Launch the "helloCuda" kernel on the device
    helloCuda<<<1, 1>>> ();

    // Wait for the kernel to finish executing
    cudaDeviceSynchronize();

    return 0;
}
