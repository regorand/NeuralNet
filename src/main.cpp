#include <iostream>
#include <cmath>

#include "nnet/NNLayer.h"
#include "nnet/NeuralNet.h"
#include "nnet/ActivationFunction.h"
#include "nnet/Utils.h"
#include "Classifiers.h"
#include "nnet/LinAlg/Matrix.h"

typedef struct dataSample {
    std::shared_ptr<NNet::Matrix> input;
    std::shared_ptr<NNet::Matrix> expectedOutput;
} Sample;

void test(bool manual);

int main() {
    //classifyIris();

    test(true);
    exit(0);


    std::shared_ptr<NNet::Matrix> mat1 = std::make_shared<NNet::Matrix>(3, 3);
    std::shared_ptr<NNet::Matrix> mat2 = std::make_shared<NNet::Matrix>(3, 1);

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

    auto res = mat1->multiply(mat2);


    /*
    mat1->print();
    mat2->print();
    mat1->applyComponentBinaryOp(mat2, [](float v1, float v2) { return v1 * v2; });

    mat1->print();
*/

    // std::shared_ptr<NNet::Matrix> res = mat1.multiply(mat2);
    // exit(0);

    /*
    classifyIris();
    exit(0);
    */

    NNet::NeuralNet net;
    std::vector<NNet::NNLayerDef> layerDefs({{2, NNet::RELU},
                                             {1, NNet::RELU}});

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

    for (size_t i = 0; i < 4; i++) {
        std::shared_ptr<NNet::Matrix> input = std::make_shared<NNet::Matrix>(2, 1);
        std::shared_ptr<NNet::Matrix> expected = std::make_shared<NNet::Matrix>(1, 1);

        float bit1 = i / 2;
        float bit2 = i % 2;

        float expectedBit = bit1 == bit2 ? 0 : 1;

        input->set(0, 0, bit1);
        input->set(1, 0, bit2);

        expected->set(0, 0, expectedBit);
        samples.push_back({input, expected});
    }

    for (size_t i = 0; i < 100; i++) {
        NNet::clearNet(net);
        float loss = 0;


        for (auto &sample: samples) {
            std::shared_ptr<NNet::Matrix> result = NNet::evaluateNet(net, sample.input);
            std::shared_ptr<NNet::Matrix> diff = std::make_shared<NNet::Matrix>(result->height(), 1);


            result->applyComponentBinaryOpInto(sample.expectedOutput,
                                               [](float result, float expected) { return result - expected; },
                                               diff);
#ifdef DEBUG
            std::cout << "\nresult: " << result[0]
                        << "\nexpected: " << sample.expectedOutput[0]
                        << "\ndiff: " << diff[0] << std::endl;
#endif
            /*
            sample.input->print();
            result->print();
            sample.expectedOutput->print();
            diff->print();
             */

            loss += 0.5f * diff->summedSquareMagnitude();

            NNet::backPropagation(net, diff);
        }
        /*
        net.layers[0].weightMatrix->print();
        net.layers[0].cumulatedGradient->print();
*/

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
        NNet::applyGradients(net, 0.1);
    }
}


void test(bool manual) {
    if (!manual) {

        NNet::NeuralNet net;
        std::vector<NNet::NNLayerDef> layerDefs({{8, NNet::SIGMOID},
                                                 {8, NNet::SIGMOID},
                                                 {1, NNet::LINEAR}});

        NNet::createNeuralNet(1, layerDefs, &net);

        std::random_device rd;  // Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
        std::uniform_real_distribution<> dis(-10.0, 10.0);
        for (size_t i = 0; i < 1; i++) {
            NNet::clearNet(net);
            float loss = 0;

            size_t num_samples = 1;
            for (size_t j = 0; j < num_samples; j++) {

                float x = dis(gen);
                //float y = dis(gen);

                std::shared_ptr<NNet::Matrix> Inp = std::make_shared<NNet::Matrix>(1, 1);
                Inp->set(0, 0, x);
                //Inp->set(1, 0, y);

                std::shared_ptr<NNet::Matrix> expected = std::make_shared<NNet::Matrix>(1, 1);
                expected->set(0, 0, x * x);

                std::shared_ptr<NNet::Matrix> result = NNet::evaluateNet(net, Inp);
                std::shared_ptr<NNet::Matrix> diff = std::make_shared<NNet::Matrix>(result->height(), 1);


                result->applyComponentBinaryOpInto(expected,
                                                   [](float result, float expected) { return result - expected; },
                                                   diff);


                /*
                std::cout << "\nX: \t\t\t" << x << std::endl;
                std::cout << "Res: \t\t" << result->get(0, 0) << std::endl;
                std::cout << "Exp: \t\t" << expected->get(0, 0) << std::endl;
                std::cout << "Diff: \t\t" << diff->get(0, 0) << std::endl;
                */
                loss += 0.5f * diff->summedSquareMagnitude();

                NNet::backPropagation(net, diff);


                //net.layers[0].cumulatedGradient->print();

            }


            std::cout << "Iteration " << i << "\t\tLoss: " << loss / num_samples << std::endl;
            std::cout << "Gradient Length " << NNet::getGradientMagnitude(net) << std::endl;

            // net.layers[0].weightMatrix->print();
            // net.layers[0].cumulatedGradient->print();
            NNet::applyGradients(net, 0.1);
        }
        printNet(net, PRINT_RESULTS_FLAG | PRINT_GRADIENTS_FLAG | PRINT_BACKPROP_FLAG);
    } else {

        NNet::NeuralNet net;
        std::vector<NNet::NNLayerDef> layerDefs({{2, NNet::RELU},
                                                 {1, NNet::LINEAR}});

        NNet::createNeuralNet(2, layerDefs, &net, 69420);

        size_t sampleCount = 1;
        std::vector<Sample> samples;
        for (size_t i = 0; i < sampleCount; i++) {
            auto s = NNet::sampleXOR(69420);
            Sample sample;
            sample.input = std::make_shared<NNet::Matrix>(2, 1);
            sample.expectedOutput = std::make_shared<NNet::Matrix>(1, 1);
            sample.input->set(0, 0, s[0]);
            sample.input->set(1, 0, s[1]);

            sample.expectedOutput->set(0, 0, s[0] == s[1] ? 0 : 1);

            samples.push_back(sample);
        }

        std::cout << "Sample Input: \n";
        samples[0].input->print();
        std::cout << "Sample Expected: \n";
        samples[0].expectedOutput->print();

        for (auto &sample: samples) {
            auto result = evaluateNet(net, sample.input);

            auto diff = std::make_shared<NNet::Matrix>(1, 1);
            std::cout << "Diff: \n";

            result->applyComponentBinaryOpInto(sample.expectedOutput,
                                               [](float v1, float v2) {return v1 - v2; },
                                               diff);



            sample.expectedOutput->applyComponentBinaryOpInto(result,
                                                              [](float v1, float v2) {return v1 - v2; },
                                                              diff);


            diff->print();
            backPropagation(net, diff);

            printNet(net, PRINT_RESULTS_FLAG | PRINT_GRADIENTS_FLAG | PRINT_BACKPROP_FLAG);
        }
    }
}