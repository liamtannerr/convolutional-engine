#include "../include/ConvolutionEngine.h"
#include <iostream>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../third_party/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../third_party/stb_image_write.h"

// --- CUDA KERNEL ---
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