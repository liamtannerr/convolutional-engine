#include "ConvolutionalEngine.h"
#include <iostream>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define TILE_SIZE 16
#define MAX_KERNEL_RADIUS 3

// --- CLASS IMPLEMENTATION ---
ConvolutionalEngine::ConvolutionalEngine() : h_inputImage(nullptr), h_outputImage(nullptr), 
    d_inputImage(nullptr), d_outputImage(nullptr), d_kernel(nullptr), width(0), height(0), channels(0), executionTime(0.0f) {}

ConvolutionalEngine::~ConvolutionalEngine() {
    freeGPUMemory();
}

// --- MEMORY AND I/O ---
bool ConvolutionalEngine::loadImage(const std::string& filename) {
    h_inputImage = stbi_load(filename.c_str(), &width, &height, &channels, 0);
    if (!h_inputImage) {
        std::cerr << "Failed to load image." << std::endl;
        return false;
    }
    
    // Allocate host memory for the 1-channel output image (Grayscale/Convolved)
    h_outputImage = new unsigned char[width * height];
    return true;
}

bool ConvolutionalEngine::saveImage(const std::string& filename) {
    if (!h_outputImage) return false;
    // Outputting as 1-channel (grayscale) PNG
    return stbi_write_png(filename.c_str(), width, height, 1, h_outputImage, width);
}

void ConvolutionalEngine::allocateGPUMemory() {
    // Allocate max potential size for input (original RGB image)
    cudaMalloc(&d_inputImage, width * height * channels * sizeof(unsigned char));
    cudaMalloc(&d_outputImage, width * height * sizeof(unsigned char));
    cudaMalloc(&d_kernel, (MAX_KERNEL_RADIUS * 2 + 1) * (MAX_KERNEL_RADIUS * 2 + 1) * sizeof(float)); 
}

void ConvolutionalEngine::freeGPUMemory() {
    if (d_inputImage) cudaFree(d_inputImage);
    if (d_outputImage) cudaFree(d_outputImage);
    if (d_kernel) cudaFree(d_kernel);
    if (h_inputImage) stbi_image_free(h_inputImage);
    if (h_outputImage) delete[] h_outputImage;
}

void ConvolutionalEngine::calculateGrid(dim3& grid, dim3& block){
    block = dim3(16, 16);
    grid = dim3((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
}

float ConvolutionalEngine::getLastExecutionTime() const {
    return executionTime;
}

// --- CUDA KERNELS ---
__global__ void grayscaleKernel(unsigned char* d_in, unsigned char* d_out, int w, int h, int chan) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w && y < h){
        int pixel_idx = y * w + x;
        int RGBidx = pixel_idx * chan;
        
        float red = d_in[RGBidx];
        float green = d_in[RGBidx + 1];
        float blue = d_in[RGBidx + 2];

        float Luminosity = 0.2126f * red + 0.7152f * green + 0.0722f * blue;
        d_out[pixel_idx] = static_cast<unsigned char>(Luminosity);
    }
}

__global__ void naiveConvolutionKernel(unsigned char* d_in, unsigned char* d_out, float* d_kern, int w, int h, int kSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int radius = kSize / 2;
    float sum = 0.0f;
    if (x < w && y < h){
        for (int i = -radius; i <= radius; i++){
            for (int j = -radius; j <= radius; j++){
                int dy = y + i;
                int dx = x + j;
                if ((dx < 0) || (dx >= w) || (dy < 0) || (dy >= h)) continue;
                sum += d_kern[(i + radius) * kSize + (j + radius)] * static_cast<float>(d_in[dy * w + dx]);
            }
        }    
        d_out[y * w + x] = static_cast<unsigned char>(fmaxf(0.0f, fminf(sum, 255.0f)));
    }
}

__global__ void tiledConvolution(unsigned char* d_in, unsigned char* d_out, float* d_kern, int w, int h, int kSize) {
    int lx = threadIdx.x;
    int ly = threadIdx.y;
    int x = blockIdx.x * TILE_SIZE + lx;
    int y = blockIdx.y * TILE_SIZE + ly;
    
    int radius = kSize / 2;
    
    __shared__ float tile[TILE_SIZE + 2 * MAX_KERNEL_RADIUS][TILE_SIZE + 2 * MAX_KERNEL_RADIUS];
     
    for (int i = ly; i < TILE_SIZE + 2 * radius; i += TILE_SIZE){
        for (int j = lx; j < TILE_SIZE + 2 * radius; j += TILE_SIZE){
            int row = blockIdx.y * TILE_SIZE + i - radius;
            int col = blockIdx.x * TILE_SIZE + j - radius;

            if (row >= 0 && row < h && col >= 0 && col < w){
                tile[i][j] = static_cast<float>(d_in[row * w + col]);
            } else {
                tile[i][j] = 0.0f;
            }
        }
    }

    __syncthreads();

    if (x < w && y < h){
        float sum = 0.0f;
        for (int i = 0; i < kSize; i++){
            for (int j = 0; j < kSize; j++){
                sum += tile[ly + i][lx + j] * d_kern[i * kSize + j];
            }
        }
        d_out[y * w + x] = static_cast<unsigned char>(fmaxf(0.0f, fminf(255.0f, sum)));
    }
}

// --- HOST WRAPPERS ---
void ConvolutionalEngine::convertToGrayscale() {
    allocateGPUMemory();
    cudaMemcpy(d_inputImage, h_inputImage, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    dim3 grid, block;
    calculateGrid(grid, block);
    
    grayscaleKernel<<<grid, block>>>(d_inputImage, d_outputImage, width, height, channels);
    cudaDeviceSynchronize();
    
    // Copy result back to CPU so we can use it as input for filtering
    cudaMemcpy(h_outputImage, d_outputImage, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
}

void ConvolutionalEngine::applyFilterNaive(const std::vector<float>& kernel, int kernelSize) {
    cudaMemcpy(d_kernel, kernel.data(), kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);
    
    // Upload the grayscaled image as the new 1-channel input
    cudaMemcpy(d_inputImage, h_outputImage, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 grid, block;
    calculateGrid(grid, block);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    naiveConvolutionKernel<<<grid, block>>>(d_inputImage, d_outputImage, d_kernel, width, height, kernelSize);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&executionTime, start, stop);

    cudaMemcpy(h_outputImage, d_outputImage, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
}

void ConvolutionalEngine::applyFilter(const std::vector<float>& kernel, int kernelSize) {
    cudaMemcpy(d_kernel, kernel.data(), kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);
    
    // Upload the grayscaled image as the new 1-channel input
    cudaMemcpy(d_inputImage, h_outputImage, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 grid, block;
    calculateGrid(grid, block);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    tiledConvolution<<<grid, block>>>(d_inputImage, d_outputImage, d_kernel, width, height, kernelSize);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&executionTime, start, stop);

    cudaMemcpy(h_outputImage, d_outputImage, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
}
