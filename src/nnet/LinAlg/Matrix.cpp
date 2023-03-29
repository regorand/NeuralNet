//
//

#include <exception>
#include <iostream>
#include <cstring>
#include "Matrix.h"

namespace NNet {
    Matrix::Matrix(size_t height, size_t width) : m_height(height), m_width(width) {
        data = new float[width * height];
    }

    Matrix::~Matrix() {
        delete data;
    }

    float Matrix::get(size_t i, size_t j) const {
        if (i >= m_height || j >= m_width) throw std::exception();
        return MAT_VAL(i, j);
    }

    void Matrix::set(size_t i, size_t j, float val) {
        if (i >= m_height || j >= m_width) throw std::exception();
        MAT_VAL(i, j) = val;
    }

    bool Matrix::canMultiply(std::shared_ptr<Matrix> other) {
        return m_width == other->m_height;
    }

    bool Matrix::sameSize(std::shared_ptr<Matrix> other) {
        return m_width == other->m_width && m_height == other->m_height;
    }

    std::shared_ptr<Matrix> Matrix::multiply(std::shared_ptr<Matrix> other) {
        if (!canMultiply(other)) throw MatrixSizeMismatchException();

        std::shared_ptr<Matrix> res = std::make_shared<Matrix>(m_height, other->m_width);

        for (size_t i = 0; i < m_height; i++) {
            for (size_t j = 0; j < other->m_width; j++) {
                float sum = 0;

                for (size_t k = 0; k < m_width; k++) {
                    sum += MAT_VAL(i, j) * other->data[k + j * m_width];
                }

                res->data[i * other->m_width + j] = sum;
            }
        }
        return res;
    }

    void Matrix::applyComponentBinaryOp(const std::shared_ptr<Matrix>& other, float (*operation)(float, float)) {
        if (!sameSize(other)) throw MatrixSizeMismatchException();

        for (size_t i = 0; i < m_height; i++) {
            for (size_t j = 0; j < m_width; j++) {
                MAT_VAL(i, j) = (*operation) (MAT_VAL(i, j), other->get(i, j));
            }
        }
    }

    std::shared_ptr<Matrix> Matrix::returnComponentBinaryOp(const std::shared_ptr<Matrix> &other, float (*operation)(float, float)) {
        if (!sameSize(other)) throw MatrixSizeMismatchException();

        auto newMat = this->copy();
        newMat->applyComponentBinaryOp(other, operation);

        return newMat;
    }

    void Matrix::print() {
        std::cout << "\n-----" << std::endl;
        for (size_t i = 0; i < m_height; i++) {
            for (size_t j = 0; j < m_width; j++) {
                std::cout << MAT_VAL(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }

    void Matrix::overwrite(float overwriteValue) {
        for (size_t i = 0; i < m_height * m_width; i++) {
            data[i] = overwriteValue;
        }
    }

    void Matrix::initRandom(std::mt19937 rng, std::uniform_real_distribution<> distribution) {
        for (size_t i = 0; i < m_height * m_width; i++) {
            data[i] = distribution(rng);
        }
    }

    std::shared_ptr<Matrix> Matrix::copy() {
        std::shared_ptr<Matrix> newMat = std::make_shared<Matrix>(m_height, m_width);

        std::memcpy(newMat->data, data, m_height * m_width * sizeof(float));

        return newMat;
    }

    float Matrix::summedSquareMagnitude() {
        float sum = 0;

        for (size_t i = 0; i < m_height * m_width; i++) {
            sum += data[i] * data[i];
        }

        return sum;
    }

    float Matrix::summedAbsMagnitude() {
        float sum = 0;

        for (size_t i = 0; i < m_height * m_width; i++) {
            sum += fabsf(data[i]);
        }

        return sum;
    }
}
