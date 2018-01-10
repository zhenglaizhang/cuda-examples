cmake_minimum_required(VERSION 3.8)
project(cuda_examples)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES add.cu)
add_executable(cuda_examples ${SOURCE_FILES})