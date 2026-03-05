#include "../include/ConvolutionalEngine.h"
#include <iostream>

int main(int argc, char** argv){

  //Ensure an image path was provided
  if (argc < 2){
    std::cerr << "Usage: ./engine <input_image_path>" << std::endl;
  }

  std::string inputPath = argv[1];
  ConvolutionalEngine engine;



  return 0;
}