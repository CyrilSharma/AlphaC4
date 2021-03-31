cmake_minimum_required(VERSION 3.17)
project(Connect_4)

set(CMAKE_CXX_STANDARD 17)

# find_library(TENSORFLOW_LIB tensorflow HINT $ENV{HOME}/libtensorflow2/lib)
find_library(TENSORFLOW_LIB tensorflow HINT usr/local/lib)

add_executable(Connect_4 main.cpp mcts.h mcts.cpp C4.h C4.cpp NeuralNetwork.cpp NeuralNetwork.h thread_pool.h)

add_executable(test_cppflow test_cppflow.cpp)

target_include_directories(test_cppflow PRIVATE /usr/local/include)
target_include_directories(test_cppflow PRIVATE cppflow/include/ usr/local/include)

target_link_libraries (test_cppflow "${TENSORFLOW_LIB}")
# target_include_directories(test_cppflow PRIVATE cppflow/include/ $ENV{HOME}/libtensorflow2/include)
target_link_libraries(test_cppflow "${TENSORFLOW_LIB}")

target_include_directories(Connect_4 PRIVATE /usr/local/include)
target_include_directories(Connect_4 PRIVATE cppflow/include/ usr/local/include)

target_link_libraries (Connect_4 "${TENSORFLOW_LIB}")
# target_include_directories(test_cppflow PRIVATE cppflow/include/ $ENV{HOME}/libtensorflow2/include)
target_link_libraries(Connect_4 "${TENSORFLOW_LIB}")

find_library(TENSORFLOW_LIB tensorflow HINT $ENV{HOME}/libtensorflow2/lib)

find_package(TensorflowCC REQUIRED)