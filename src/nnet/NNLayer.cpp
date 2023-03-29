//
//

#include <iostream>
#include "NNLayer.h"

namespace NNet {

    void calculateLayerOutput(std::vector<float> &inputVector, NNLayer &layer) {
        layer.layerOutput.clear();
        layer.preAccOutput.clear();
        layer.layerOutput.resize(layer.weightMatrix.size());
        layer.preAccOutput.resize(layer.weightMatrix.size());
        for (size_t N_idx = 0; N_idx < layer.weightMatrix.size(); N_idx++) {
            if (inputVector.size() != layer.weightMatrix[N_idx].size()) {
                std::cerr << "Layer size doesnt match input size" << std::endl;
                throw std::exception();
            }

            float sum = layer.bias[N_idx];

            for (size_t M_idx = 0; M_idx < layer.weightMatrix[N_idx].size(); M_idx++) {
                sum += layer.weightMatrix[N_idx][M_idx] * inputVector[M_idx];
            }
            layer.preAccOutput[N_idx] = sum;
            layer.layerOutput[N_idx] = (*(layer.activationFunction.activation)) (sum);
        }
    }

    size_t layerOuputSize(NNLayer layer) { return layer.weightMatrix.size(); };
    size_t layerInputSize(NNLayer layer) { return layer.weightMatrix[0].size(); };

}