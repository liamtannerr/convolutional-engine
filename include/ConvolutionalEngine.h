#ifndef CONVOLUTION_ENGINE_H
#define CONVOLUTION_ENGINE_H

#include <string>
#include <vector>

/**
 * @brief A modular C++/CUDA library for high-performance image filtering.
 */
class ConvolutionalEngine{

  public:
    ConvolutionalEngine();
    ~ConvolutionalEngine();

    // ---- Image I/O and Basic Processing ----
    bool loadImage(const std::string& filename);
    bool saveImage(const std::string& filename);

    // ---- Grayscaling ----
    void convertToGrayscale();

    // ---- Convolutional Operations ----
    void applyFilterNaive(const std::vector<float>& kernel, int kernelSize);
    void applyFilter(const std::vector<float>& kernel, int kernelSize);

    // ---- Benchmarking ----
    float getLastExecutionTime() const;
 
  private:
    unsigned char* h_inputImage;
    unsigned char* h_outputImage;

    unsigned char* d_inputImage;
    unsigned char* d_outputImage;
    float* d_kernel;

    int width, height, channels;
    float executionTime; // Added to store benchmark times

    // Fixed signature to match implementation (grid first, block second)
    void calculateGrid(dim3& grid, dim3& block);

    void allocateGPUMemory();
    void freeGPUMemory();
};

#endif