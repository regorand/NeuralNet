//
//

#ifndef NEURALNET_ACTIVATIONFUNCTION_H
#define NEURALNET_ACTIVATIONFUNCTION_H

#include <string>

namespace NNet {

    typedef struct activation_func_s {
        float (*activation)(float val);
        float (*derivative)(float val);
        std::string name;
    } ActivationFunction;

    float linear(float val);
    float linearDerivative(float val);

    float sigmoid(float val);
    float sigmoidDerivative(float val);

    float tanh(float val);
    float tanhDerivative(float val);

    float ReLU(float val);
    float ReLUDerivative(float val);

    static const ActivationFunction LINEAR = {&linear, &linearDerivative, "Linear"};
    static const ActivationFunction SIGMOID = {&sigmoid, &sigmoidDerivative, "Sigmoid"};
    static const ActivationFunction TANH = {&tanh, &tanhDerivative, "Tanh"};
    static const ActivationFunction RELU = {&ReLU, &ReLUDerivative, "ReLU"};
}


#endif //NEURALNET_ACTIVATIONFUNCTION_H
