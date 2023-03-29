//
//

#include "ActivationFunction.h"
#include <cmath>

namespace NNet {

    float linear(float val){
        return val;
    }

    float linearDerivative(float val) {
        return 1;
    }

    float sigmoid(float val) {
        return 1 / (1 + exp(-val));
    }

    float sigmoidDerivative(float val) {
        return sigmoid(val) * (1 - sigmoid(val));
    }

    float tanh(float val) {
        return std::tanh(val);
    }

    float tanhDerivative(float val) {
        float tanhValue = tanh(val);
        return 1 - tanhValue * tanhValue;
    }

    float ReLU(float val) {
        return val > 0 ? val : 0;
    }

    float ReLUDerivative(float val) {
        return val > 0 ? 1 : 0;
    }
}