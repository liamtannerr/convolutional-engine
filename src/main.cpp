#include "ConvolutionalEngine.h"
#include <iostream>
#include <vector>

int main(int argc, char** argv){

    if (argc < 2){
        std::cerr << "Usage: ./engine <input_image_path>" << std::endl;
        return -1;
    }

    std::string inputPath = argv[1];
    ConvolutionalEngine engine;

    std::cout << "Loading image: " << inputPath << "..." << std::endl;
    if (!engine.loadImage(inputPath)) return -1;

    // Step 1: Convert to Grayscale
    std::cout << "Converting to Grayscale..." << std::endl;
    engine.convertToGrayscale();
    engine.saveImage("output_gray.png");

    // 3x3 Edge Detection Kernel
    std::vector<float> edgeKernel = {
        -1.0f, -1.0f, -1.0f,
        -1.0f,  8.0f, -1.0f,
        -1.0f, -1.0f, -1.0f
    };
    int kSize = 3;

    // Step 2: Test Naive Convolution
    std::cout << "\nRunning Naive Convolution..." << std::endl;
    engine.applyFilterNaive(edgeKernel, kSize);
    std::cout << "Naive Execution Time: " << engine.getLastExecutionTime() << " ms" << std::endl;
    engine.saveImage("output_naive_edges.png");

    // Resetting grayscale base so we don't apply edges on top of edges
    engine.convertToGrayscale();

    // Step 3: Test Tiled (Optimized) Convolution
    std::cout << "\nRunning Tiled Shared Memory Convolution..." << std::endl;
    engine.applyFilter(edgeKernel, kSize);
    std::cout << "Tiled Execution Time: " << engine.getLastExecutionTime() << " ms" << std::endl;
    engine.saveImage("output_tiled_edges.png");

    std::cout << "\nSuccess! Check your directory for output files." << std::endl;

    return 0;
}