#include "../include/ConvolutionalEngine.h"
#include <iostream>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../third_party/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../third_party/stb_image_write.h"

/*
Global Variables for Tiled Convolution
*/
// Always use Tile Size of 16
#define TILE_SIZE 16
// Support up to 7x7 Convolution
#define MAX_KERNEL_RADIUS 3

// --- CLASS IMPLEMENTATION ---
ConvolutionalEngine::ConvolutionalEngine() : h_inputImage(nullptr), h_outputImage(nullptr), 
    d_inputImage(nullptr), d_outputImage(nullptr), width(0), height(0), channels(0) {}

ConvolutionalEngine::~ConvolutionalEngine() {
    freeGPUMemory();
}

// --- GRID AND BLOCK SIZE HELPER ---

void ConvolutionalEngine::calculateGrid(dim3& block, dim3& grid){
  //Initialize a thread block of size 256
  block = dim3(16, 16);
  // Calculate the number of thread blocks needed to cover the image
  grid = dim3((block.x + width - 1) / block.x, (block.y + height - 1)/ block.y);
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
    d_out[pixel_idx] = static_cast<unsigned char>(Luminosity);

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

  /*
    Tiled convolution with tile size 16
  */
  __global__ void tiledConvolution(unsigned char* d_in, unsigned char* d_out,
  float* d_kern, int w, int h, int kSize) {

    // Local x and y coordinates
    int lx = threadIdx.x;
    int ly = threadIdx.y;

    // Global position
    int x = blockIdx.x * TILE_SIZE + lx;
    int y = blockIdx.y * TILE_SIZE + ly;
    
    int radius = kSize / 2;
    
    __shared__ float tile[TILE_SIZE + 2 * MAX_KERNEL_RADIUS][TILE_SIZE + 2 * MAX_KERNEL_RADIUS];
     
    // Load tile into shared memory including it's halo
    for (int i = ly; i < TILE_SIZE + 2 * radius; i += TILE_SIZE){
      for (int j = lx; j < TILE_SIZE + 2 * radius; j += TILE_SIZE){

        // Global row and column position
        int row = blockIdx.y * TILE_SIZE + i - radius;
        int col = blockIdx.x * TILE_SIZE + j - radius;

        if (row >= 0 && row < h && col >= 0 && col < w){
          // Map input image pixel to shared memory tile
          tile[i][j] = static_cast<float>(d_in[row * w + col]);
        } else {
          // Zero Padding
          tile[i][j] = 0.0f;
        }
      }
    }

    // Ensure the tile + halo is loaded entirely before calculations begin
    __syncthreads();

    // Compute the convolution using the shared memory tile
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



