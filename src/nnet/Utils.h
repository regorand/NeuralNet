//
//

#ifndef NEURALNET_UTILS_H
#define NEURALNET_UTILS_H

#include <vector>

namespace NNet {

    void doMatrixVectorMult(std::vector<std::vector<float>> &matrix, std::vector<float> &vector, std::vector<float> &result);

    void initMatrixUniformly(std::vector<std::vector<float>> &matrix);

    void initArrayUniformly(std::vector<float> &vector);

    void doVectorVectorTransposeToMatrix(std::vector<float> &vector1, std::vector<float> &vector2, std::vector<std::vector<float>> &resultMat);

    std::vector<float> sampleXOR();

}

#endif //NEURALNET_UTILS_H
