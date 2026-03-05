#ifndef CONVOLUTION_ENGINE_H
#define CONVOLUTION_ENGINE_H

#include <string>
#include <vector>


/**

 * @brief A modular C++/CUDA library for high-performance image filtering.

 */
class ConvolutionalEngine{

  public:
    // Allocate and Deallocate resources with a Constructor and Destructor
    ConvolutionalEngine();
    ~ConvolutionalEngine();

    // ---- Image I/O and Basic Processing ----

    // Load the image using stb_image
    bool loadImage(const std::string& filename);

    // Save the image using stb_image_write
    bool saveImage(const std::string& filename);

    // ---- Grayscaling ----

    // Pointwise image Grayscaling
    void convertToGrayscale();

    // ---- Convolutional Operations ----

    // Naive global gemory convolution
    void applyFilterNaive(const std::vector<float>& kernel, int kernelSize);

    // Optimized, tiled, shared memory convolution
    void applyFilter(const std::vector<float>& kernel, int kernelSize);

    // ---- Benchmarking ----

    // Apply timing logic to compare naive vs optimized convolutions
    float getLastExecutionTime() const;
 
  private:

    // Pointers to data on the CPU (Host)
    unsigned char* h_inputImage;
    unsigned char* h_outputImage;

    // Pointers to data on the GPU (Device)
    unsigned char* d_inputImage;
    unsigned char* d_outputImage;
    float* d_kernel;

    int width, height, channels;

    // Internal helper to manage GPU memory allocation/deallocation 
    void allocateGPUMemory();
    void freeGPUMemory();


};

#endif