cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80 86 89)
endif()

# Avoid mixing static and shared CUDA runtimes in the same binary.
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

project(nano_vllm_cpp CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)

# On WSL, a native-Linux libcuda.so installed by the CUDA toolkit can shadow
# the WSL driver stub (/usr/lib/wsl/lib/libcuda.so.1).  LD_LIBRARY_PATH is
# the only mechanism that reliably overrides the search for all dlopen calls.
set(NANO_VLLM_WSL FALSE)
if(EXISTS "/usr/lib/wsl/lib/libcuda.so.1" AND EXISTS "/lib/x86_64-linux-gnu/libcuda.so.1")
    set(NANO_VLLM_WSL TRUE)
    message(STATUS "WSL detected: will set LD_LIBRARY_PATH=/usr/lib/wsl/lib for tests")
endif()

# ---- Include paths ----
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/third_party)

# ---- xxhash ----
find_library(XXHASH_LIB xxhash REQUIRED)

# ---- PCRE2 (for tokenizer regex) ----
find_path(PCRE2_INCLUDE_DIR pcre2.h)
find_library(PCRE2_LIBRARY pcre2-8)
if(PCRE2_INCLUDE_DIR AND PCRE2_LIBRARY)
    message(STATUS "PCRE2 found: ${PCRE2_LIBRARY}")
else()
    message(FATAL_ERROR "PCRE2 not found. Install libpcre2-dev.")
endif()

# ---- NCCL (optional) ----
find_path(NCCL_INCLUDE_DIR nccl.h
    HINTS
        /usr/include
        /usr/include/x86_64-linux-gnu
        /usr/local/cuda/include)
find_library(NCCL_LIBRARY nccl
    HINTS
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        /usr/local/cuda/lib64)

set(NANO_VLLM_USE_NCCL 0)
if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
    set(NANO_VLLM_USE_NCCL 1)
    message(STATUS "NCCL support enabled: ${NCCL_LIBRARY}")
else()
    message(STATUS "NCCL support disabled: nccl.h or libnccl not found")
endif()

# ---- Library: nano_vllm_core (纯 CPU 部分) ----
add_library(nano_vllm_core
    src/core/config.cpp
    src/engine/sequence.cpp
    src/engine/block_manager.cpp
    src/engine/model_runner.cpp
    src/engine/scheduler.cpp
    src/kernels/kernel_types.cu
    src/kernels/rms_norm.cu
    src/kernels/rotary.cu
    src/kernels/activation.cu
    src/kernels/bias_add.cu
    src/kernels/gather.cu
    src/kernels/embedding.cu
    src/kernels/sampling.cu
    src/kernels/kvcache.cu
    src/kernels/split_qkv.cu
    src/kernels/attention.cu
    src/kernels/flash_attn.cpp
    src/layers/activation.cpp
    src/layers/attention.cpp
    src/layers/embed_head.cpp
    src/layers/layernorm.cpp
    src/layers/linear.cpp
    src/layers/rotary_embedding.cpp
    src/layers/sampler.cpp
    src/models/transformer.cpp
    src/utils/cuda_allocator.cpp
    src/utils/cuda_common.cpp
    src/utils/cublas_wrapper.cpp
    src/utils/context.cpp
    src/utils/loader.cpp
    src/utils/tensor.cpp
    src/utils/tensor_parallel.cpp
    src/utils/tokenizer.cpp
    src/engine/llm_engine.cpp
)
set_target_properties(nano_vllm_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(nano_vllm_core PUBLIC NANO_VLLM_USE_NCCL=${NANO_VLLM_USE_NCCL})
if(NANO_VLLM_USE_NCCL)
    target_include_directories(nano_vllm_core PUBLIC ${NCCL_INCLUDE_DIR})
endif()
target_link_libraries(nano_vllm_core PUBLIC ${XXHASH_LIB} ${PCRE2_LIBRARY} CUDA::cudart CUDA::cublas CUDA::cublasLt dl)
if(PCRE2_INCLUDE_DIR)
    target_include_directories(nano_vllm_core PUBLIC ${PCRE2_INCLUDE_DIR})
endif()
if(NANO_VLLM_USE_NCCL)
    target_link_libraries(nano_vllm_core PUBLIC ${NCCL_LIBRARY})
endif()

# ---- GoogleTest ----
enable_testing()
find_package(GTest REQUIRED)

add_executable(test_config tests/test_config.cpp)
target_link_libraries(test_config PRIVATE nano_vllm_core GTest::gtest GTest::gtest_main)
add_test(NAME test_config COMMAND test_config)

add_executable(test_sequence tests/test_sequence.cpp)
target_link_libraries(test_sequence PRIVATE nano_vllm_core GTest::gtest GTest::gtest_main)
add_test(NAME test_sequence COMMAND test_sequence)

add_executable(test_block_manager tests/test_block_manager.cpp)
target_link_libraries(test_block_manager PRIVATE nano_vllm_core GTest::gtest GTest::gtest_main)
add_test(NAME test_block_manager COMMAND test_block_manager)

add_executable(test_scheduler tests/test_scheduler.cpp)
target_link_libraries(test_scheduler PRIVATE nano_vllm_core GTest::gtest GTest::gtest_main)
add_test(NAME test_scheduler COMMAND test_scheduler)

add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE nano_vllm_core GTest::gtest GTest::gtest_main)
add_test(NAME test_tensor COMMAND test_tensor)

add_executable(test_layers tests/test_layers.cpp)
target_link_libraries(test_layers PRIVATE nano_vllm_core GTest::gtest GTest::gtest_main)
add_test(NAME test_layers COMMAND test_layers)

add_executable(test_loader tests/test_loader.cpp)
target_link_libraries(test_loader PRIVATE nano_vllm_core GTest::gtest GTest::gtest_main)
add_test(NAME test_loader COMMAND test_loader)

add_executable(test_models tests/test_models.cpp)
target_link_libraries(test_models PRIVATE nano_vllm_core GTest::gtest GTest::gtest_main)
add_test(NAME test_models COMMAND test_models)

add_executable(test_model_runner tests/test_model_runner.cpp)
target_link_libraries(test_model_runner PRIVATE nano_vllm_core GTest::gtest GTest::gtest_main)
add_test(NAME test_model_runner COMMAND test_model_runner)

add_executable(test_llm_engine tests/test_llm_engine.cpp)
target_link_libraries(test_llm_engine PRIVATE nano_vllm_core GTest::gtest GTest::gtest_main)
add_test(NAME test_llm_engine COMMAND test_llm_engine)

add_executable(bench_runtime_layers bench/bench_runtime_layers.cpp)
target_link_libraries(bench_runtime_layers PRIVATE nano_vllm_core)

add_executable(bench_kernels bench/bench_kernels.cpp)
target_link_libraries(bench_kernels PRIVATE nano_vllm_core)

add_executable(run_qwen3_once bench/run_qwen3_once.cpp)
target_link_libraries(run_qwen3_once PRIVATE nano_vllm_core)

add_executable(example example.cpp)
target_link_libraries(example PRIVATE nano_vllm_core)

add_executable(bench_e2e bench/bench_e2e.cpp)
target_link_libraries(bench_e2e PRIVATE nano_vllm_core)

add_executable(bench bench.cpp)
target_link_libraries(bench PRIVATE nano_vllm_core)

# WSL: set LD_LIBRARY_PATH for all CTest runs so the correct libcuda is found.
if(NANO_VLLM_WSL)
    set(_wsl_env "LD_LIBRARY_PATH=/usr/lib/wsl/lib:$ENV{LD_LIBRARY_PATH}")
    set_tests_properties(
        test_config test_sequence test_block_manager test_scheduler
        test_tensor test_layers test_loader test_models
        test_model_runner test_llm_engine
        PROPERTIES ENVIRONMENT "${_wsl_env}"
    )
endif()
