//
//
#ifndef NEURALNET_NNLAYER_H
#define NEURALNET_NNLAYER_H

#include <inttypes.h>
#include <vector>
#include "ActivationFunction.h"

namespace NNet {

    typedef struct layer_s {
        std::vector<std::vector<float>> weightMatrix; // Matrix of m_width numberInputs, m_height numberNodes, calculate output as multiplication with input vector
        std::vector<float> bias;
        std::vector<float> layerOutput;
        std::vector<float> preAccOutput;
        ActivationFunction activationFunction;

        std::vector<float> cumulatedBiasGradient;
        std::vector<std::vector<float>> cumulatedWeightGradient;
        uint32_t countGradients;
    } NNLayer;

    typedef struct layer_def_s {
        uint32_t layerSize;
        ActivationFunction activationFunction;
    } NNLayerDef;

    void calculateLayerOutput(std::vector<float> &inputVector, NNLayer &layer);

    size_t layerOuputSize(NNLayer layer);
    size_t layerInputSize(NNLayer layer);

}

#endif //NEURALNET_NNLAYER_H
