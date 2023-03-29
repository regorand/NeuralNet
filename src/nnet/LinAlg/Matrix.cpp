//
//

#include <exception>
#include <iostream>
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
        return data[i + j * m_height];
    }

    void Matrix::set(size_t i, size_t j, float val) {
        if (i >= m_height || j >= m_width) throw std::exception();
        data[i + j * m_height] = val;
    }

    std::shared_ptr<Matrix> Matrix::multiply(Matrix &other) {
        if (m_width != other.m_height) throw std::exception();

        std::shared_ptr<Matrix> res = std::make_shared<Matrix>(m_height, other.m_width);

        for (size_t i = 0; i < m_height; i++) {
            for (size_t j = 0; j < other.m_width; j++) {
                float sum = 0;

                for (size_t k = 0; k < m_width; k++) {
                    sum += data[i + k * m_height] * other.data[k + j * m_width];
                }

                res->data[i * other.m_width + j] = sum;
            }
        }
        return res;
    }

    void Matrix::print() {
        std::cout << "\n-----" << std::endl;
        for (size_t i = 0; i < m_height; i++) {
            for (size_t j = 0; j < m_width; j++) {
                std::cout << data[i + j * m_height] << " ";
            }
            std::cout << std::endl;
        }
    }
}