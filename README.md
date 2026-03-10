# CUDA Convolution Engine

A modular C++/CUDA library for high-performance image filtering. This project demonstrates the transition from standard CPU-based image manipulation to a massively parallel GPU-accelerated pipeline.



## Current Project Status
The project is currently in the Core Kernel Development phase. The architecture focuses on decoupling high-level C++ application logic from low-level CUDA hardware execution.

### Completed
* **Architectural Scaffolding**: Established professional C++ filesystem and Public API (ConvolutionEngine.h).
* **Phase 1 (Infrastructure & Grayscale)**: Implemented host-to-device memory orchestration and a pointwise luminosity kernel.
* **Phase 2 (Naive Convolution)**: Developed a 2D stencil kernel utilizing a one-thread-per-pixel mapping strategy with zero-padding boundary conditions.

### In Progress / Pending
* **Build System**: Configuring a Makefile for multi-stage compilation using nvcc (sm_75) and g++.
* **Phase 3 (Tiled Optimization)**: Implementing Shared Memory (SRAM) tiling to reduce global memory bottlenecks and maximize cache hits.
* **Phase 4 (Performance Benchmarking)**: Adding high-resolution timers to compare GPU throughput vs. CPU execution.

## File Breakdown
* **include/ConvolutionEngine.h**: The Public API. Uses Object-Oriented patterns to hide VRAM management from the user.
* **src/main.cpp**: The Driver. Handles CLI arguments and demonstrates library usage for end-users.
* **src/ConvolutionEngine.cu**: The Implementation. Contains the CUDA kernels and private memory management logic.



## Engineering Highlights
* **RAII Management**: Class Destructors ensure deterministic cudaFree calls, preventing VRAM leaks during long-running tasks (critical for robotics/AUVIC).
* **2D Thread Mapping**: Optimized 16x16 block configurations to maximize occupancy on NVIDIA T4 and Jetson hardware.
* **Type Safety**: Utilizing static_cast and explicit floating-point literals (0.0f) to optimize for 32-bit hardware execution units.
