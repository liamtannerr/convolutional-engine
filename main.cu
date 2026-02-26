#include <iostream>
#include <cuda_runtime.h>

// A simple "Hello from GPU" kernel
__global__ void helloFromGPU() {
    printf("Hello from Thread %d in Block %d!\n", threadIdx.x, blockIdx.x);
}

int main() {
    std::cout << "Starting CUDA Starter Project..." << std::endl;

    // Launch the kernel with 1 block and 5 threads
    helloFromGPU<<<1, 5>>>();

    // Wait for the GPU to finish before the CPU continues
    cudaDeviceSynchronize();

    std::cout << "GPU task complete. Environment is ready for Computer Vision!" << std::endl;
    return 0;
}
