//
//

#ifndef NEURALNET_NEURALNET_H
#define NEURALNET_NEURALNET_H

#include <vector>
#include "NNLayer.h"

namespace NNet {

    typedef struct neural_net_d {
        std::vector<NNLayer> layers;
        std::vector<float> lastInputs;
    } NeuralNet;

    typedef struct neural_net_eval_d {
        NeuralNet net;
        std::vector<float> lastInputs;
    } NeuralNetEval;

    std::vector<float> evaluateNet(NeuralNet &net, std::vector<float> &input);

    void createNeuralNet(uint32_t inputSize, const std::vector<NNLayerDef> &layerDefs, NeuralNet *net);

    void backPropagation(NeuralNet &net, std::vector<float> &outputDiffs, bool softMax = false, float softmaxDenom = 1.0f);

    void applyGradients(NeuralNet &net, float learningRate);

    void clearNet(NeuralNet &net);

    float getGradientMagnitude(NeuralNet &net);
}

#endif //NEURALNET_NEURALNET_H
