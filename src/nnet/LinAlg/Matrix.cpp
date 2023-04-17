//
//

#include <exception>
#include <iostream>
#include <cstring>
#include <sstream>
#include <iomanip>

#include "Matrix.h"

namespace NNet {
    Matrix::Matrix(size_t height, size_t width) : m_height(height), m_width(width), m_isTransposed(false) {
        data = new float[width * height];
    }

    Matrix::~Matrix() {
        delete data;
    }

    float Matrix::get(size_t i, size_t j) const {
        if (i >= m_height || j >= m_width) throw std::exception();
        return _get(i, j);
    }

    void Matrix::set(size_t i, size_t j, float val) {
        if (i >= m_height || j >= m_width) throw std::exception();
        MAT_VAL(i, j) = val;
    }

    bool Matrix::canMultiply(const std::shared_ptr<Matrix>& other) const {
        return m_width == other->m_height;
    }

    bool Matrix::sameSize(const std::shared_ptr<Matrix>& other) const {
        return m_width == other->m_width && m_height == other->m_height;
    }

    void Matrix::multiplyInto(const std::shared_ptr<Matrix>& other, const std::shared_ptr<Matrix>& dest) {
        if (!canMultiply(other)) throw MatrixSizeMismatchException();
        if (m_height != dest->m_height || other->m_width != dest->m_width) throw MatrixSizeMismatchException();

        for (size_t i = 0; i < m_height; i++) {
            for (size_t j = 0; j < other->m_width; j++) {
                float sum = 0;

                for (size_t k = 0; k < m_width; k++) {
                    sum += MAT_VAL(i, k) * other->_get(k, j);
                    // sum += MAT_VAL(i, j) * other->data[k + j * m_width];
                }

                dest->data[i * other->m_width + j] = sum;
            }
        }
    }

    std::shared_ptr<Matrix> Matrix::multiply(const std::shared_ptr<Matrix>& other) {
        if (!canMultiply(other)) throw MatrixSizeMismatchException();

        std::shared_ptr<Matrix> res = std::make_shared<Matrix>(m_height, other->m_width);
        multiplyInto(other, res);

        return res;
    }

    /*
    template<typename Lambda>
    void Matrix::applyComponentBinaryOp(const std::shared_ptr<Matrix>& other, Lambda&& operation) {
        if (!sameSize(other)) throw MatrixSizeMismatchException();

        for (size_t i = 0; i < m_height; i++) {
            for (size_t j = 0; j < m_width; j++) {
                MAT_VAL(i, j) = operation(MAT_VAL(i, j), other->get(i, j));
            }
        }
    }
    */

    /*
    template<typename Lambda>
    void Matrix::applyComponentBinaryOpInto(const std::shared_ptr<Matrix> &other, Lambda &&operation,
                                            const std::shared_ptr<Matrix> &dest) {
        if (!sameSize(other)) throw MatrixSizeMismatchException();
        if (!sameSize(dest)) throw MatrixSizeMismatchException();

        for (size_t i = 0; i < m_height; i++) {
            for (size_t j = 0; j < m_width; j++) {
                dest->set(i, j, operation(MAT_VAL(i, j), other->get(i, j)));
            }
        }
    }
    */

    template<typename Lambda>
    std::shared_ptr<Matrix> Matrix::returnComponentBinaryOp(const std::shared_ptr<Matrix> &other, Lambda &&operation) {
        if (!sameSize(other)) throw MatrixSizeMismatchException();

        auto newMat = this->copy();
        newMat->applyComponentBinaryOp(other, operation);

        return newMat;
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

    void Matrix::initFromStdVector(std::vector<float> &vec) {
        size_t maxElements = std::min(m_height * m_width, vec.size());

        std::memcpy(data, vec.data(), maxElements);
    }

    float Matrix::_get(size_t i, size_t j) const {
        return MAT_VAL(i, j);
    }

    bool Matrix::isTransposed() {
        return m_isTransposed;
    }

    void Matrix::transpose() {
        m_isTransposed = !m_isTransposed;

        size_t tmp = m_height;
        m_height = m_width;
        m_width = tmp;
    }

    std::shared_ptr<Matrix> Matrix::copyAndAddBiasTerm() {
        if (m_width != 1) throw MatrixSizeMismatchException();

        std::shared_ptr<Matrix> res = std::make_shared<Matrix>(m_height + 1, m_width);

        std::memcpy(res->data, data, m_height * sizeof(float));
        res->set(m_height, 0, 1);
        return res;
    }

    void Matrix::initFromUntil(const std::shared_ptr<Matrix> &src, size_t count) {
        if (src->m_width != 1) throw MatrixSizeMismatchException();
        if (src->m_height < count) throw MatrixSizeMismatchException();
        if (m_height < count) throw MatrixSizeMismatchException();

        std::memcpy(data, src->data, count * sizeof(float));
    }

    /*
    template<typename Lambda>
    void Matrix::mapFrom(const std::shared_ptr<Matrix>& src, Lambda &&map) {
        if (!sameSize(src)) throw MatrixSizeMismatchException();
        std::transform(src->data, src->data + (m_height * m_width), data, map);
    }
     */

    void Matrix::print() {
        std::cout << asString() << std::endl;
    }

    std::string Matrix::asString(int indent, int width, int precision) {
        std::stringstream stream;
        std::stringstream indentStream;
        for (size_t i = 0; i < indent; i++) {
            indentStream << "\t";
        }
        std::string indentStr = indentStream.str();
        std::string lineBreak = "\n" + indentStr;

        stream.precision(3);
        stream << std::fixed;
        stream << indentStr << "[\t";
        for (size_t i = 0; i < m_height; i++) {
            for (size_t j = 0; j < m_width; j++) {
                stream << std::setw(width) << MAT_VAL(i, j) << " ";
            }
            if (i != m_height - 1){
                stream << lineBreak;
            }
            stream << "\t";
        }
        stream << "]";

        return stream.str();
    }

    template<typename Lambda>
    void Matrix::mapInplace(Lambda &&map) {
        std::transform(data, data + (m_height * m_width), data, map);
    }
}
