//
//
#ifndef NEURALNET_NNLAYER_H
#define NEURALNET_NNLAYER_H

#include <inttypes.h>
#include <vector>
#include "ActivationFunction.h"
#include "LinAlg/Matrix.h"

namespace NNet {
    typedef struct {
        std::shared_ptr<Matrix> weightMatrix;
        ActivationFunction activationFunction;

        std::shared_ptr<Matrix> layerOutput;
        std::shared_ptr<Matrix> preAccOutput;
        std::shared_ptr<Matrix> internalError;

        std::shared_ptr<Matrix> cumulatedGradient;
    } NNLayer;

    typedef struct layer_def_s {
        uint32_t layerSize;
        ActivationFunction activationFunction;
    } NNLayerDef;
}

#endif //NEURALNET_NNLAYER_H
