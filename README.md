# CUDA Convolution Engine

A modular C++/CUDA library for high-performance image filtering. This project transitions from standard CPU-based image manipulation to a massively parallel GPU-accelerated pipeline.



## Current Project Status
The project is currently in the **Architectural Scaffolding** phase. The system follows a decoupled library design to separate the user interface from the hardware-specific kernels.

### Completed
* **Project Directory Structure**: Established a professional C++ filesystem (`include/`, `src/`, `third_party/`).
* **Public API Design**: Defined the `ConvolutionEngine.h` header, establishing the contract for image I/O and filtering phases.
* **Driver Baseline**: Implemented a rudimentary `main.cpp` to handle command-line arguments and orchestrate library calls.
* **Dependency Integration**: Integrated `stb_image` for lightweight, cross-platform image loading.

### In Progress / Pending
* **Phase 1 Implementation**: Writing the `ConvolutionEngine.cu` logic for host-to-device memory transfers and the Grayscale kernel.
* **Build System**: Configuring the `Makefile` to link NVCC and G++ objects.
* **Phase 2-4 Kernels**: Development of naive and optimized (Shared Memory) convolution logic.

## File breakdown
* **`include/ConvolutionEngine.h`**: The Public API. It encapsulates GPU complexity within a clean C++ class.
* **`src/main.cpp`**: The Driver. An entry point for users to interact with the library via the CLI.
* **`src/ConvolutionEngine.cu`**: The Implementation (Pending). Where CUDA kernels and VRAM management live.



## Engineering Highlights
* **RAII Management**: Utilizing C++ Destructors to ensure `cudaFree` is called automatically, preventing VRAM leaks.
* **Separation of Concerns**: Decoupling image loading, memory allocation, and mathematical kernels for maximum reusability.
* **SIMT Optimization**: Designing for a 2D grid/block architecture to map threads directly to pixel coordinates.
