cmake_minimum_required(VERSION 2.4)
project(NeuralNetwork)
set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(PROJECT_CODE
                               include/NeuralNetwork/matrix.hpp
                               include/NeuralNetwork/costFunctionStrategy.hpp
  src/crossEntropyCost.cpp     include/NeuralNetwork/crossEntropyCost.hpp
  src/meanSquereErrorCost.cpp  include/NeuralNetwork/meanSquereErrorCost.hpp
  src/image.cpp                include/NeuralNetwork/image.hpp
  src/layer.cpp                include/NeuralNetwork/layer.hpp
  src/mnistDataLoader.cpp      include/NeuralNetwork/mnistDataLoader.hpp
  src/neuralnetwork.cpp        include/NeuralNetwork/neuralnetwork.hpp
  src/reluLayer.cpp            include/NeuralNetwork/reluLayer.hpp
  src/sigmoidLayer.cpp         include/NeuralNetwork/sigmoidLayer.hpp
  src/userInterface.cpp        include/NeuralNetwork/userInterface.hpp)

set(CATCH2_SRC
  include/catch2/catch_reporter_automake.hpp
  include/catch2/catch_reporter_tap.hpp
  include/catch2/catch_reporter_teamcity.hpp
  include/catch2/catch.hpp)

include_directories(include)
include_directories(include/NeuralNetwork)

add_executable(NeuralNetwork
  ${PROJECT_CODE} src/main.cpp)

add_executable(Tests
  ${PROJECT_CODE} ${CATCH2_SRC} src/tests.cpp)