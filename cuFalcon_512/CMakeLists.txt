cmake_minimum_required(VERSION 3.22)
project(cuFalcon CUDA CXX C)

# =============================
# Global configuration
# =============================
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CUDA_ARCHITECTURES 89)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# =============================
# Common source files
# =============================
set(SOURCES
        src/rng.cu
        src/shake.cu
        src/sign.cu
        src/ffSampling.cu
        src/common.cu
        src/fft.cu
        src/expanded_key.cu
        src/api.cu
)

# Main entry file
set(MAIN_FILE main.cu)

# =============================
# cuFalcon512 executables
# =============================
add_executable(cuFalcon512_opt      ${MAIN_FILE} ${SOURCES})
add_executable(cuFalcon512_tree     ${MAIN_FILE} ${SOURCES})
add_executable(cuFalcon512_balance  ${MAIN_FILE} ${SOURCES})

# Enable separable compilation
foreach(target cuFalcon512_opt cuFalcon512_tree cuFalcon512_balance)
    set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endforeach()

# Add compile definitions
target_compile_definitions(cuFalcon512_opt     PRIVATE TARGET_STREAM_OPT)
target_compile_definitions(cuFalcon512_tree    PRIVATE TARGET_STREAM_TREE)
target_compile_definitions(cuFalcon512_balance PRIVATE TARGET_STREAM_BALANCE)

