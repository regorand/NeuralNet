//
//

#ifndef NEURALNET_NEURALNET_H
#define NEURALNET_NEURALNET_H

#include <vector>
#include "NNLayer.h"
#include "LinAlg/Matrix.h"

namespace NNet {

    typedef struct neural_net_d {
        std::vector<NNLayer> layers;
        std::shared_ptr<Matrix> lastInputs; // column vector, with 1 appended as for bias
        uint32_t countGradients;
    } NeuralNet;

    typedef struct neural_net_eval_d {
        NeuralNet net;
        std::vector<float> lastInputs;
    } NeuralNetEval;

    std::shared_ptr<Matrix> evaluateNet(NeuralNet &net, const std::shared_ptr<Matrix>& input);

    void createNeuralNet(uint32_t inputSize, const std::vector<NNLayerDef> &layerDefs, NeuralNet *net, uint32_t seed = time(nullptr));

    void backPropagation(NeuralNet &net, const std::shared_ptr<Matrix>& outputDiffs);

    void applyGradients(NeuralNet &net, float learningRate);

    void clearNet(NeuralNet &net);

    float getGradientMagnitude(NeuralNet &net);

#define PRINT_RESULTS_FLAG 1
#define PRINT_GRADIENTS_FLAG 2
#define PRINT_BACKPROP_FLAG 4

    void printNet(NeuralNet &net, int flags = 0);
}

#endif //NEURALNET_NEURALNET_H
