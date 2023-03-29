//
//

#include "Utils.h"
#include <random>
#include <iostream>

namespace NNet {

    void doMatrixVectorMult(std::vector<std::vector<float>> &matrix, std::vector<float> &vector, std::vector<float> &result) {
        for (size_t mat_row_idx = 0; mat_row_idx < matrix.size(); mat_row_idx++) {
            if (vector.size() != matrix[mat_row_idx].size()) {
                std::cerr << "Layer size doesnt match input size" << std::endl;
                throw std::exception();
            }

            float sum = 0;

            for (size_t mat_col_idx = 0; mat_col_idx < matrix[mat_row_idx].size(); mat_col_idx++) {
                sum += matrix[mat_row_idx][mat_col_idx] * vector[mat_col_idx];
            }

            result[mat_row_idx] = sum;
        }
    }

    void initMatrixUniformly(std::vector<std::vector<float>> &matrix) {
        for (auto &v: matrix) {
            initArrayUniformly(v);
        }
    }

    void initArrayUniformly(std::vector<float> &vector) {
        std::random_device rd;  // Will be used to obtain a seed for the random number engine

        std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
        std::uniform_real_distribution<> dis(-1.0, 1.0);


        for (size_t i = 0; i < vector.size(); i++) {
            vector[i] = dis(gen);
        }
    }

    void doVectorVectorTransposeToMatrix(std::vector<float> &vector1, std::vector<float> &vector2, std::vector<std::vector<float>> &resultMat) {
        resultMat.resize(vector2.size());
        for (auto &mat: resultMat) mat.resize(vector1.size());

        for (size_t i = 0; i < vector2.size(); i++) {
            for (size_t j = 0; j < vector1.size(); j++) {
                resultMat[i][j] = vector1[j] * vector2[i];
            }
        }
    }

    std::vector<float> sampleXOR() {
        std::random_device rd;  // Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
        std::uniform_real_distribution<> dis(0, 1.0);

        std::vector<float> res;
        res.push_back(dis(gen) > 0.5 ? 1.0 : 0.0);
        res.push_back(dis(gen) > 0.5 ? 1.0 : 0.0);
        return res;
    }

}