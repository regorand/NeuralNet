//
//

#include "NeuralNet.h"
#include <iostream>
#include "Utils.h"

#include <cmath>
#include <cstring>
#include <sstream>
#include <iomanip>


namespace NNet {

    std::shared_ptr<Matrix> evaluateNet(NeuralNet &net, const std::shared_ptr<Matrix>& input) {
        if (net.layers.empty()) {
            std::cerr << "Net needs layers" << std::endl;
            throw std::exception();
        }
        if (input->height() + 1 != net.layers[0].weightMatrix->width()) {
            std::cerr << "First Layer input size doesn't match input vector size" << std::endl;
            throw std::exception();
        }

        net.lastInputs = input;
        std::shared_ptr<Matrix> layerInputs = input->copyAndAddBiasTerm();

        for (size_t i = 0; i < net.layers.size(); i++) {
            // for (auto &layer : net.layers) {
            NNLayer layer = net.layers[i];

            if (!layer.weightMatrix->canMultiply(layerInputs)) throw MatrixSizeMismatchException();

            layer.weightMatrix->multiplyInto(layerInputs, layer.preAccOutput);
            layer.layerOutput->mapFrom(layer.preAccOutput, layer.activationFunction.activation);

            layerInputs = layer.layerOutput->copyAndAddBiasTerm();
        }

        return net.layers[net.layers.size() - 1].layerOutput;
    }

    void createNeuralNet(uint32_t inputSize, const std::vector<NNLayerDef> &layerDefs, NeuralNet *net, uint32_t seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<> dis(-1.0, 1.0);

        net->lastInputs = std::make_shared<Matrix>(inputSize + 1, 1);
        net->lastInputs->overwrite(0);

        net->layers.clear();
        uint32_t prevLayerSize = inputSize;
        for (auto &layerDef: layerDefs) {
            NNLayer layer;

            layer.activationFunction = layerDef.activationFunction;

            layer.weightMatrix = std::make_shared<Matrix>(layerDef.layerSize, prevLayerSize +
                                                                              1); // Add 1 because biases are  part of weight matrix
            layer.weightMatrix->initRandom(rng, dis);

            layer.layerOutput = std::make_shared<Matrix>(layerDef.layerSize, 1);
            layer.layerOutput->overwrite(0);

            layer.preAccOutput = std::make_shared<Matrix>(layerDef.layerSize, 1);
            layer.preAccOutput->overwrite(0);

            layer.internalError = std::make_shared<Matrix>(prevLayerSize, 1);
            layer.internalError->overwrite(0);

            layer.cumulatedGradient = std::make_shared<Matrix>(layerDef.layerSize, prevLayerSize + 1);
            layer.cumulatedGradient->overwrite(0);

            prevLayerSize = layerDef.layerSize;
            net->layers.push_back(layer);
        }
    }

    void backPropagation(NeuralNet &net, const std::shared_ptr<Matrix>& outputDiffs) {
        // TODO: temporarily removed softmax function, maybe find more general solution afterwards for output transformations ?

        for (int32_t i = net.layers.size() - 1; i >= 0; i--) {
            NNLayer layer = net.layers[i];
            std::shared_ptr<Matrix> dPreAc = std::make_shared<Matrix>(layer.layerOutput->height(),
                                                                      layer.layerOutput->width());

            // 1. Calculate derivative: layOutput -> preAcc

            auto derivative = layer.activationFunction.derivative;
            auto operation = [derivative](float outputDiff, float outputVal) {
                return -outputDiff * derivative(outputVal);
            };

            if (i == net.layers.size() - 1) {
                outputDiffs->applyComponentBinaryOpInto(layer.preAccOutput, operation, dPreAc);
            } else {
                net.layers[i + 1].internalError->applyComponentBinaryOpInto(layer.preAccOutput, operation, dPreAc);
                //layer.internalError->applyComponentBinaryOpInto(layer.layerOutput, operation, dPreAc);
            }


            // 2. Calculate derivate: preAcc -> weightMat
            std::shared_ptr<Matrix> prevInput;
            if (i > 0) {
                prevInput = net.layers[i - 1].layerOutput;
            } else {
                prevInput = net.lastInputs;
            }

            // dPreAc->transpose();
            auto extended = prevInput->copyAndAddBiasTerm();
            std::shared_ptr<Matrix> dWeightMat = std::make_shared<Matrix>(dPreAc->height(), extended->height());
            extended->transpose();
            dPreAc->multiplyInto(extended, dWeightMat);

            // 3. Calculate derivative: preAcc -> prevOutput
            // Only calculate derivative to previous layer output if there is a previous layer
            if (i > 0) {
                std::shared_ptr<Matrix> tmpInternalError =
                        std::make_shared<Matrix>(net.layers[i - 1].layerOutput->height() + 1, 1);

                // TODO: net yet sure if next line is correct: To be tested: is bias correctly calculated out ?
                layer.weightMatrix->transpose();
                layer.weightMatrix->multiplyInto(dPreAc, tmpInternalError);
                layer.weightMatrix->transpose();

                layer.internalError->initFromUntil(tmpInternalError, layer.internalError->height());
            }

            // 4. Save dWeightMat into cumulatedGradient
            layer.cumulatedGradient->applyComponentBinaryOp(dWeightMat, [](float v1, float v2) {return v1 + v2; });
            // -- Old:

            /*
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
                doVectorVectorTransposeToMatrix(net.layers[i - 1].layerOutput, dPA, dWeightMat);
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

            for (size_t j = 0; j < net.layers[i].bias.size(); j++) {
                net.layers[i].cumulatedBiasGradient[j] += dPA[j];
            }
            for (size_t j = 0; j < net.layers[i].weightMatrix.size(); j++) {
                for (size_t k = 0; k < net.layers[i].weightMatrix[j].size(); k++) {
                    net.layers[i].cumulatedWeightGradient[j][k] += dWeightMat[j][k];
                }
            }

            net.layers[i].countGradients++;
             */
        }
        net.countGradients++;
    }

    void applyGradients(NeuralNet &net, float learningRate) {
        float gradientMagnitude = getGradientMagnitude(net);

        gradientMagnitude = gradientMagnitude == 0 ? 1 : gradientMagnitude; //prevents division by 0

        gradientMagnitude = 1;

        float countFactor = 1.0f / (net.countGradients);

        auto applyGradientOp = [countFactor, learningRate](float prevVal, float gradientVal) {
            return prevVal - countFactor * learningRate * gradientVal;
        };

        for (auto &layer: net.layers) {
            layer.weightMatrix->applyComponentBinaryOp(layer.cumulatedGradient, applyGradientOp);
        }
    }

    void clearNet(NeuralNet &net) {
        net.countGradients = 0;
        for (auto &layer: net.layers) {
            layer.cumulatedGradient->overwrite(0);

            // These are probably not needed, they are here for clarity for now, test and remove later
            layer.layerOutput->overwrite(0);
            layer.preAccOutput->overwrite(0);
            layer.internalError->overwrite(0);
        }
    }

    float getGradientMagnitude(NeuralNet &net) {
        double sum = 0;
        size_t total = 0;
        for (auto &layer: net.layers) {
            float layerMag = layer.cumulatedGradient->summedSquareMagnitude();
            sum += layerMag;
            total += layer.cumulatedGradient->height() * layer.cumulatedGradient->width();
        }
        return sqrt(sum / static_cast<double>(total));
    }

    void printNet(NeuralNet &net, int flags) {
        bool printInternalResults = flags & PRINT_RESULTS_FLAG;
        bool printGradient = flags & PRINT_GRADIENTS_FLAG;
        bool printBackProp = flags & PRINT_BACKPROP_FLAG;

        std::stringstream stream;
        stream << std::setprecision(6);

        stream << "Printing Net:\n\n";

        if (printGradient) {
            stream << "Gradient Count: " << net.countGradients << "\n\n";
        }

        if (printInternalResults) {
            stream << "Last Input: \n";
            net.lastInputs->transpose();
            stream << net.lastInputs->asString();
            net.lastInputs->transpose();
            stream << "\n\n";
        }

        for (size_t i = 0; i < net.layers.size(); i++) {
            auto layer = net.layers[i];
            stream << "Layer: " << i << "\n";
            auto width = layer.weightMatrix->width() - 1;
            auto height = layer.weightMatrix->height();
            stream << "\t"
                    << width
                    << " Input"
                    << ((width > 1) ? "s\t\t" : "\t\t")
                    << layer.weightMatrix->height() << " Output"
                    << ((height > 1) ? "s\n" : "\n");
            stream << "\tActivation Function: " << layer.activationFunction.name << "\n\n";

            stream << "\tWeight Matrix: \n" << layer.weightMatrix->asString(2) << "\n\n";

            if (printInternalResults) {

                stream << "\tPre Activation Output: \n"
                        << layer.preAccOutput->asString(2)
                        << "\n\n"
                        << "\tLayer output: \n"
                        << layer.layerOutput->asString(2)
                        << "\n\n";
            }

            if (printBackProp) {
                stream << "\tInternal Error: \n"
                       << layer.internalError->asString(2)
                       << "\n\n";
            }

            if (printGradient) {
                stream << "\tCumulated Gradient: \n"
                       << layer.cumulatedGradient->asString(2)
                       << "\n\n";
            }
        }
        std::cout << stream.str() << std::endl;
    }
}