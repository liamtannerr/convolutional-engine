# CUDA Convolution Engine

A modular C++/CUDA library for high-performance image filtering. This project demonstrates the transition from standard CPU-based image manipulation to a massively parallel GPU-accelerated pipeline.

## Current Project Status
The project is currently in the Performance Benchmarking phase. The architecture focuses on decoupling high-level C++ application logic from low-level CUDA hardware execution.

### Completed
* **Architectural Scaffolding**: Established professional C++ filesystem and Public API (ConvolutionalEngine.h).
* **Build System**: Implemented a unified Makefile for multi-stage compilation using nvcc and g++.
* **Phase 1 (Infrastructure & Grayscale)**: Implemented host-to-device memory orchestration and a pointwise luminosity kernel.
* **Phase 2 (Naive Convolution)**: Developed a 2D stencil kernel utilizing a one-thread-per-pixel mapping strategy.
* **Phase 3 (Tiled Optimization)**: Implemented Shared Memory (SRAM) tiling with halo/apron loading to reduce global memory bottlenecks and maximize cache hits.

### In Progress / Pending
* **Phase 4 (Performance Benchmarking)**: Adding high-resolution timers to compare GPU throughput vs. CPU execution across different kernel sizes.

## File Breakdown
* **include/ConvolutionalEngine.h**: The Public API. Uses Object-Oriented patterns to hide VRAM management from the user.
* **src/main.cpp**: The Driver. Handles CLI arguments and orchestrates library calls.
* **src/ConvolutionalEngine.cu**: The Implementation. Contains CUDA kernels, shared memory logic, and VRAM management.
* **Makefile**: Automates the compilation and linking of CUDA and C++ source files.

## Engineering Highlights
* **RAII Management**: Class Destructors ensure deterministic cudaFree calls, preventing VRAM leaks during long-running tasks.
* **Shared Memory Tiling**: Optimized data reuse by loading image tiles into SRAM, reducing redundant global memory accesses.
* **2D Thread Mapping**: Configured 16x16 block dimensions to maximize occupancy on NVIDIA hardware.
* **Type Safety**: Utilizing static_cast and explicit floating-point literals (0.0f) to optimize for 32-bit hardware execution units.



## Engineering Highlights
* **RAII Management**: Class Destructors ensure deterministic cudaFree calls, preventing VRAM leaks during long-running tasks (critical for robotics/AUVIC).
* **2D Thread Mapping**: Optimized 16x16 block configurations to maximize occupancy on NVIDIA T4 and Jetson hardware.
* **Type Safety**: Utilizing static_cast and explicit floating-point literals (0.0f) to optimize for 32-bit hardware execution units.
