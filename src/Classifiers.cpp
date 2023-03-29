//
//
#include "Classifiers.h"
#include "nnet/NeuralNet.h"

#include <filesystem>
#include <fstream>
#include <vector>
#include <iostream>
#include <sstream>
#include <random>


void classifyIris() {
    std::fstream fs;
    fs.open("../DataSets/Iris/iris.data", std::fstream::in);

    std::vector<IrisSample> samples;
    std::string line;

    std::getline(fs, line);
    char dummy;
    while (!line.empty()) {
        std::stringstream ss(line);
        IrisSample sample;
        ss >> sample.sepalLength;
        ss >> dummy;
        ss >> sample.sepalWidth;
        ss >> dummy;
        ss >> sample.petalLength;
        ss >> dummy;
        ss >> sample.petalWidth;
        ss >> dummy;
        std::string name;
        ss >> name;
        if (name == "Iris-setosa") sample.irisClass = IrisClass::SETOSA;
        if (name == "Iris-versicolor") sample.irisClass = IrisClass::VERSICOLOUR;
        if (name == "Iris-virginica") sample.irisClass = IrisClass::VIRGINICA;

        samples.push_back(sample);
        std::getline(fs, line);
    }

    auto rng = std::default_random_engine{};
    std::shuffle(samples.begin(), samples.end(), rng);

    const float trainSetFactor = 0.3;

    std::vector<IrisSample> trainSet;
    std::vector<IrisSample> testSet;

    const size_t trainSetSize = trainSetFactor * samples.size();
    const size_t testSetSize = samples.size() - trainSetSize;
    for (size_t i = 0; i < samples.size(); i++) {
        if (i < trainSetSize) {
            trainSet.push_back(samples[i]);
        } else {
            testSet.push_back(samples[i]);
        }
    }

    NNet::NeuralNet net;

    std::vector<NNet::NNLayerDef> layerDefs({{4, NNet::SIGMOID},
                                             {3, NNet::SIGMOID}});

    NNet::createNeuralNet(4, layerDefs, &net);

    float learningRate = 1;
    float prevLoss = 100000;
    for (size_t i = 0; i < 10000; i++) {
        NNet::clearNet(net);
        float loss = 0;

        std::shuffle(trainSet.begin(), trainSet.end(), rng);


        size_t trainCount = 0;
        size_t trainCorrect = 0;
        for (auto &sample: trainSet) {
            std::vector<float> input({sample.sepalLength, sample.sepalWidth, sample.petalLength, sample.petalWidth});
            std::vector<float> result = NNet::evaluateNet(net, input);

            float softmaxNormalization = 0;
            for (float k: result) {
                softmaxNormalization += exp(k);
            }

            std::vector<float> softmaxValues;

            softmaxValues.push_back(exp(result[0]) / softmaxNormalization);
            softmaxValues.push_back(exp(result[1]) / softmaxNormalization);
            softmaxValues.push_back(exp(result[2]) / softmaxNormalization);

            std::vector<float> diff(3);

            size_t maxIndex = -1;
            float maxVal = -100;
            for (size_t j = 0; j < softmaxValues.size(); j++) {
                if (result[j] > maxVal) {
                    maxVal = result[j];
                    maxIndex = j;
                }

                diff[j] = result[j] - (static_cast<int>(sample.irisClass) == j ? 1.0f : 0.0f);
                loss += 0.5f * diff[j] * diff[j];
            }

            if ((static_cast<int>(sample.irisClass) == maxIndex)) {
                trainCorrect++;
            }
            trainCount++;

            NNet::backPropagation(net, diff);
        }

        float testLoss = 0;
        size_t testCount = 0;
        size_t testCorrect = 0;
        for (auto &sample: testSet) {
            std::vector<float> input({sample.sepalLength, sample.sepalWidth, sample.petalLength, sample.petalWidth});
            std::vector<float> result = NNet::evaluateNet(net, input);

            size_t maxIndex = -1;
            float maxVal = -100;
            for (size_t j = 0; j < result.size(); j++) {
                if (result[j] > maxVal) {
                    maxVal = result[j];
                    maxIndex = j;
                }
                float diff = result[j] - (static_cast<int>(sample.irisClass) == j ? 1.0f : 0.0f);
                testLoss += 0.5f * diff * diff;
            }
            if ((static_cast<int>(sample.irisClass) == maxIndex)) {
                testCorrect++;
            }
            testCount++;
        }

        if (i % 1 == 0) {
            std::cout << "Iteration: " << i
                      << "\tLoss: " << loss / trainSetSize
                      << "\tTrain Accuracy: " << ((float) trainCorrect) / trainCount
                      << "\tTest Loss: " << testLoss / testSetSize
                      << "\tTest Accuracy: " << ((float) testCorrect) / testCount
                      //<< "\tLearning Rate: " << learningRate
                      << std::endl;
        }
        // std::cout << "Gradient Length " << NNet::getGradientMagnitude(net) << std::endl;


        if (loss > prevLoss) {
            learningRate -= 0.001;
            // if (learningRate <= 0) break;
        }
        prevLoss = loss;
        learningRate = loss / trainSetSize;
        NNet::applyGradients(net, learningRate);
    }
}
