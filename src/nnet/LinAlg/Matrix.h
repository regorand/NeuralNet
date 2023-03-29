//
//

#ifndef NEURALNET_MATRIX_H
#define NEURALNET_MATRIX_H


#include <cstddef>
#include <memory>
#include <random>

#define MAT_VAL(i, j) (data[i + j * m_height])

namespace NNet {

    /**
     * This Class represents a Matrix, the most important mathematical object in this library
     * This is also used to represent column and row vectors
     */
    class MatrixSizeMismatchException : public std::exception {
    public:
        char * what () {
            return "Matrix operation could be performed due to size mismatch";
        }
    };

    class Matrix {

    private:
        float *data;

        size_t m_height;
        size_t m_width;

    public:
        Matrix(size_t width, size_t height);

        ~Matrix();

        [[nodiscard]] float get(size_t i, size_t j) const;

        void set(size_t i, size_t j, float val);

        bool canMultiply(std::shared_ptr<Matrix> other);
        bool sameSize(std::shared_ptr<Matrix> other);

        std::shared_ptr<Matrix> multiply(std::shared_ptr<Matrix> other);

        // TODO: if this needs to return a new matrix, write function that creates copies (maybe also still need to write function for that) mat1, perfoms this function on it and returns it.
        void applyComponentBinaryOp(const std::shared_ptr<Matrix>& other, float (*operation)(float, float));
        std::shared_ptr<Matrix> returnComponentBinaryOp(const std::shared_ptr<Matrix>& other, float (*operation)(float, float));

        size_t width() const { return m_width; };
        size_t height() const { return m_height; };

        void overwrite(float overwriteValue);
        void initRandom(std::mt19937 rng, std::uniform_real_distribution<> distribution);

        std::shared_ptr<Matrix> copy();

        float summedSquareMagnitude();
        float summedAbsMagnitude();

        void print();


    };

}

#endif //NEURALNET_MATRIX_H
