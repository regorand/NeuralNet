cmake_minimum_required(VERSION 3.23)
project(NeuralNet)

set(CMAKE_CXX_STANDARD 20)

add_executable(NeuralNet src/main.cpp src/nnet/NNLayer.h src/nnet/NeuralNet.cpp src/nnet/NeuralNet.h src/nnet/ActivationFunction.cpp src/nnet/ActivationFunction.h src/nnet/Utils.cpp src/nnet/Utils.h src/Classifiers.cpp src/Classifiers.h src/nnet/LinAlg/Matrix.cpp src/nnet/LinAlg/Matrix.h)
