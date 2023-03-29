#include <iostream>
#include <cmath>

#include "nnet/NNLayer.h"
#include "nnet/NeuralNet.h"
#include "nnet/ActivationFunction.h"
#include "nnet/Utils.h"
#include "Classifiers.h"
#include "nnet/LinAlg/Matrix.h"

typedef struct dataSample {
    std::vector<float> input;
    std::vector<float> expectedOutput;
} Sample;

int main() {
    std::shared_ptr<NNet::Matrix> mat1 = std::make_shared<NNet::Matrix>(2, 3);
    std::shared_ptr<NNet::Matrix> mat2 = std::make_shared<NNet::Matrix>(2, 3);

    for (size_t i = 0; i < mat1->height(); i++) {
        for (size_t j = 0; j < mat1->width(); j++) {
            mat1->set(i, j, i + j);
        }
    }

    for (size_t i = 0; i < mat2->height(); i++) {
        for (size_t j = 0; j < mat2->width(); j++) {
            mat2->set(i, j, 2 * (i + j));
        }
    }

    mat1->print();
    mat2->print();
    mat1->applyComponentBinaryOp(mat2, [](float v1, float v2) { return v1 * v2; });

    mat1->print();

    // std::shared_ptr<NNet::Matrix> res = mat1.multiply(mat2);

    exit(0);

    classifyIris();
    exit(0);


    NNet::NeuralNet net;
    std::vector<NNet::NNLayerDef> layerDefs( {{2, NNet::SIGMOID}, {1, NNet::SIGMOID}});

    NNet::createNeuralNet(2, layerDefs, &net);

#ifdef DEBUG
    for (auto &layer: net.layers) {
        std::cout << "Bias: ";
        for (auto &b: layer.bias) {
            std::cout << " " << b;
        }
        std::cout << "\nWeights";
        for (auto &outer: layer.weightMatrix) {
            for (auto &w: outer) {
                std::cout << " " << w;
            }
            std::cout << "\n";
        }
        std::cout << "\n---\n";
    }
    std::cout << "\n\nSamples:" << std::endl;
#endif

    std::vector<Sample> samples;

    samples.push_back({std::vector<float>({0, 0}) , std::vector<float>({0})});
    samples.push_back({std::vector<float>({0, 1}) , std::vector<float>({1})});
    samples.push_back({std::vector<float>({1, 0}) , std::vector<float>({1})});
    samples.push_back({std::vector<float>({1, 1}) , std::vector<float>({0})});


    for (size_t i = 0; i < 300; i++) {
        NNet::clearNet(net);
        float loss = 0;

        for (auto &sample: samples) {
            auto result = NNet::evaluateNet(net, sample.input);
            std::vector<float> diff(1);

            diff[0] = result[0] - sample.expectedOutput[0];
#ifdef DEBUG
            std::cout << "\nresult: " << result[0]
                        << "\nexpected: " << sample.expectedOutput[0]
                        << "\ndiff: " << diff[0] << std::endl;
#endif
            loss += 0.5f * diff[0] * diff[0];

            NNet::backPropagation(net, diff);
        }
        std::cout << "Iteration " << i << "\t\tLoss: " << loss << std::endl;
        std::cout << "Gradient Length " << NNet::getGradientMagnitude(net) << std::endl;

#ifdef DEBUG
        for (auto &layer: net.layers) {
            std::cout << "Count: " << layer.countGradients << "\n";
            std::cout << "Bias: ";
            for (auto &b: layer.cumulatedBiasGradient) {
                std::cout << " " << b;
            }
            std::cout << "\nWeights";
            for (auto &outer: layer.cumulatedWeightGradient) {
                for (auto &w: outer) {
                    std::cout << " " << w;
                }
                std::cout << "\n";
            }
            std::cout << "\n---\n";
        }
        std::cout << std::endl;
#endif
        NNet::applyGradients(net, 1);
    }
}