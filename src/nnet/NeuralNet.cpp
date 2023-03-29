//
//

#include "NeuralNet.h"
#include <iostream>
#include "Utils.h"

#include <math.h>

namespace NNet {

    std::vector<float> evaluateNet(NeuralNet &net, std::vector<float> &input) {
        if (net.layers.empty()) {
            std::cerr << "Net needs layers" << std::endl;
            throw std::exception();
        }
        // TODO: validate nets in future if (!net.valid) throw std::exception();
        if (layerInputSize(net.layers[0]) != input.size()) {
            std::cerr << "First Layer input size doesn't match input vector size" << std::endl;
            throw std::exception();
        }

        net.lastInputs = input;
        std::vector<float> layerRes = input;
        for (auto & layer : net.layers) {
            calculateLayerOutput(layerRes, layer);
            layerRes = layer.layerOutput;
        }
        return layerRes;
    }

    void createNeuralNet(uint32_t inputSize, const std::vector<NNLayerDef> &layerDefs, NeuralNet *net) {
        net->layers.clear();
        uint32_t prevLayerSize = inputSize;
        for (auto &layerDef: layerDefs) {
            NNLayer layer;

            layer.activationFunction = layerDef.activationFunction;
            layer.bias.resize(layerDef.layerSize);
            layer.weightMatrix.resize(layerDef.layerSize);
            for (auto &row: layer.weightMatrix) {
                row.resize(prevLayerSize);
            }

            initMatrixUniformly(layer.weightMatrix);
            initArrayUniformly(layer.bias);

            prevLayerSize = layerDef.layerSize;
            net->layers.push_back(layer);
        }
    }

    void backPropagation(NeuralNet &net, std::vector<float> &outputDiffs, bool softmax, float softmaxDenom) {
        std::vector<float> internalError;

        for (int32_t i = net.layers.size() - 1; i >= 0; i--) {
            // 1. Calculate derivative of prev Layer Loss -> preActivation
            std::vector<float> dPA(net.layers[i].layerOutput.size());

            for (size_t j = 0; j < net.layers[i].layerOutput.size(); j++) {
                float a_factor = 0;
                if (i == net.layers.size() - 1) {
                    if (softmax) {
                        a_factor = exp(outputDiffs[j]) / softmaxDenom;
                    } else {
                        a_factor = outputDiffs[j];
                    }
                } else {
                    internalError[j];
                }
#ifdef DEBUG
                std::cout << "a_factor " << a_factor;
                std::cout << "\npreAccOutput " << net.layers[i].preAccOutput[j];
                std::cout << "\nderiv " << net.layers[i].activationFunction.derivative(net.layers[i].preAccOutput[j]);
                std::cout << "\n----" << std::endl;
#endif
                dPA[j] = a_factor * net.layers[i].activationFunction.derivative(net.layers[i].preAccOutput[j]);
            }

            std::vector<std::vector<float>> dWeightMat;

            if (i > 0) {
                doVectorVectorTransposeToMatrix(net.layers[i-1].layerOutput, dPA, dWeightMat);
            } else {
                doVectorVectorTransposeToMatrix(net.lastInputs, dPA, dWeightMat);
            }

            internalError.resize(layerInputSize(net.layers[i]));

            for (size_t M_idx = 0; M_idx < net.layers[i].weightMatrix[0].size(); M_idx++) {
                float sum = 0;

                for (size_t N_idx = 0; N_idx < net.layers[i].weightMatrix.size(); N_idx++) {
                    sum += net.layers[i].weightMatrix[N_idx][M_idx] * dPA[N_idx];
                }

                internalError[M_idx] = sum;
            }

            for(size_t j = 0; j < net.layers[i].bias.size(); j++) {
                net.layers[i].cumulatedBiasGradient[j] += dPA[j];
            }
            for(size_t j = 0; j < net.layers[i].weightMatrix.size(); j++) {
                for (size_t k = 0; k < net.layers[i].weightMatrix[j].size(); k++) {
                    net.layers[i].cumulatedWeightGradient[j][k] += dWeightMat[j][k];
                }
            }
            net.layers[i].countGradients++;
        }
    }

    void applyGradients(NeuralNet &net, float learningRate) {
        float gradientMagnitude = getGradientMagnitude(net);

        gradientMagnitude = gradientMagnitude == 0 ? 1 : gradientMagnitude; //prevents division by 0

        for (auto &layer: net.layers) {
            float countFactor = 1.0 / (layer.countGradients * gradientMagnitude);
            for (size_t i = 0; i < layer.bias.size(); i++) {
                layer.bias[i] -= countFactor * learningRate * layer.cumulatedBiasGradient[i];
            }
            for (size_t i = 0; i < layer.weightMatrix.size(); i++) {
                for (size_t j = 0; j < layer.weightMatrix[i].size(); j++) {
                    layer.weightMatrix[i][j] -= countFactor * learningRate * layer.cumulatedWeightGradient[i][j];
                }
            }
        }
    }

    void clearNet(NeuralNet &net) {
        for (auto &layer: net.layers) {
            layer.cumulatedBiasGradient.clear();
            layer.cumulatedBiasGradient.resize(layer.bias.size());

            layer.cumulatedWeightGradient.clear();
            layer.cumulatedWeightGradient.resize(layer.weightMatrix.size());

            for (size_t i = 0; i < layer.weightMatrix.size(); i++) {
                layer.cumulatedWeightGradient[i].clear();
                layer.cumulatedWeightGradient[i].resize(layer.weightMatrix[i].size());
            }
            layer.countGradients = 0;
        }
    }

    float getGradientMagnitude(NeuralNet &net) {
        float sum = 0;
        for(auto &layer: net.layers) {
            float countFactor = 1.0f / ((float) layer.countGradients);
            for(size_t j = 0; j < layer.cumulatedBiasGradient.size(); j++) {
                if (abs(layer.cumulatedBiasGradient[j]) >= 1) {
                    auto x = 5;
                }
                sum += countFactor * layer.cumulatedBiasGradient[j] * countFactor * layer.cumulatedBiasGradient[j];
            }

            for(size_t j = 0; j < layer.cumulatedWeightGradient.size(); j++) {
                for(size_t k = 0; k < layer.cumulatedWeightGradient[j].size(); k++) {
                    if (abs(countFactor * layer.cumulatedWeightGradient[j][k]) >= 1.0) {
                        auto x = 5;
                    }
                    sum += countFactor * layer.cumulatedWeightGradient[j][k] * countFactor * layer.cumulatedWeightGradient[j][k];
                }
            }
        }
        return sqrt(sum);
    }
}