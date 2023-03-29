//
//

#ifndef NEURALNET_CLASSIFIERS_H
#define NEURALNET_CLASSIFIERS_H

enum class IrisClass{
    SETOSA = 0,
    VERSICOLOUR = 1,
    VIRGINICA = 2
};

typedef struct {
    float sepalLength;
    float sepalWidth;
    float petalLength;
    float petalWidth;
    IrisClass irisClass;
} IrisSample;

void classifyIris();

#endif //NEURALNET_CLASSIFIERS_H
