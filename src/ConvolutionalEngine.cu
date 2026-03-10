#include "../include/ConvolutionEngine.h"
#include <iostream>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../third_party/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../third_party/stb_image_write.h"


// --- CLASS IMPLEMENTATION ---
ConvolutionEngine::ConvolutionEngine() : h_inputImage(nullptr), h_outputImage(nullptr), 
    d_inputImage(nullptr), d_outputImage(nullptr), width(0), height(0), channels(0) {}

ConvolutionEngine::~ConvolutionEngine() {
    freeGPUMemory();
}

// --- GRID AND BLOCK SIZE HELPER ---

void ConvolutionalEngine::calculateGrid(dim3& block, dim3& grid){
  //Initialize a thread block of size 256
  block = dim3(16, 16);
  // Calculate the number of thread blocks needed to cover the image
  grid = dim3((block.x + width - 1) / block.x, (block.y + width - 1)/ block.y);
}


// --- CUDA KERNELS ---
// Pointwise Luminosity. Every thread handles one pixel independently.
__global__ void grayscaleKernel(unsigned char* d_in, unsigned char* d_out, int w, int h, int chan) {

  // Get the x and y components of the thread
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < w && y < h){
    // Compute pixel address from x and y
    int pixel_idx = y * w + x;
    int RGBidx = pixel_idx * chan;
    float red = d_in[RGBidx];
    // Green and Blue are simply the next two addresses
    float green = d_in[RGBidx + 1];
    float blue = d_in[RGBidx + 2];

    float Luminosity = 0.2126 * red + 0.7152 * green + 0.0722 * blue;

    // Recast Luminosity value at compile time
    d_out[pixel_idx] = static_cast<unsigned char>Luminosity;

  }
}

__global__ void naiveConvolutionKernel(unsigned char* d_in, unsigned char* d_out, 
float* d_kern, int w, int h, int kSize) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int radius = kSize / 2;
    float sum = 0.0f;
    if (x < w && y < h){
      for (int i = -radius; i <= radius; i++){
        for (int j = -radius; j <= radius; j++){
          // Calculate offsets
          int dy = y + i;
          int dx = x + j;
          // Don't go past the image border
          if ((dx < 0) || (dx >= w) || (dy < 0) || (dy >= h)){
            continue;
          }
          // Get the pointwise sum of the current pixel and kernel weight
          sum += d_kern[(i + radius) * kSize + (j + radius)] * static_cast<float>(d_in[dy * w + dx]);
        }
      }    
      d_out[y * w + x] = static_cast<unsigned char>(fmaxf(0.0f, fminf(sum, 255.0f)));
    }

  }



