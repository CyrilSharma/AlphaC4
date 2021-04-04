cmake_minimum_required(VERSION 3.17)

project(bindings)

set(CMAKE_CXX_STANDARD 17)

set(SOURCE_DIR "src/utils")

find_library(TENSORFLOW_LIB tensorflow HINT usr/local/lib)

include_directories(lib/eigen-3.3.9)

include_directories(lib/cppflow/include)

include_directories(/usr/local/include)

include_directories(${SOURCE_DIR})

link_libraries("${TENSORFLOW_LIB}")

add_subdirectory(lib/pybind11)

set(SOURCES "${SOURCE_DIR}/mcts.cpp" "${SOURCE_DIR}/C4.cpp" "${SOURCE_DIR}/NeuralNetwork.cpp")

set(TESTS "src/C4_Test.cpp")

pybind11_add_module(bindings ${SOURCES} "${SOURCE_DIR}/bindings.cpp")

add_executable(tests ${TESTS} ${SOURCES})