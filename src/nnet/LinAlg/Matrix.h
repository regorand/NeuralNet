//
//

#ifndef NEURALNET_MATRIX_H
#define NEURALNET_MATRIX_H


#include <cstddef>
#include <memory>
#include <random>

// TODO expand this for transposing matrices
#define MAT_VAL(i, j) (m_isTransposed ? data[j + i * m_width] : data[i + j * m_height])

namespace NNet {

    /**
     * This Class represents a Matrix, the most important mathematical object for this Neural Network implementation
     * This is also used to represent column and row vectors
     */
    class MatrixSizeMismatchException : public std::exception {
    public:
        const char * what () {
            return "Matrix operation could be performed due to size mismatch";
        }
    };

    class Matrix {

    private:
        // should always access this through get() or _get()
        float *data;

        size_t m_height;
        size_t m_width;

        bool m_isTransposed;

    public:
        Matrix(size_t height, size_t width);

        ~Matrix();

        // Common accessor functions
        [[nodiscard]] float get(size_t i, size_t j) const;
        void set(size_t i, size_t j, float val);

        [[nodiscard]] size_t width() const { return m_width; };
        [[nodiscard]] size_t height() const { return m_height; };

        float summedSquareMagnitude();
        float summedAbsMagnitude();

        bool isTransposed();

        // checks for correct states or compatibility with other matrices
        [[nodiscard]] bool canMultiply(const std::shared_ptr<Matrix>& other) const;

        [[nodiscard]] bool sameSize(const std::shared_ptr<Matrix>& other) const;

        // Matrix Operations
        void transpose();

        std::shared_ptr<Matrix> multiply(const std::shared_ptr<Matrix>& other);
        void multiplyInto(const std::shared_ptr<Matrix>& other, const std::shared_ptr<Matrix>& dest);

        template<typename Lambda>
        void applyComponentBinaryOp(const std::shared_ptr<Matrix>& other, Lambda &&operation) {
            if (!sameSize(other)) throw MatrixSizeMismatchException();

            for (size_t i = 0; i < m_height; i++) {
                for (size_t j = 0; j < m_width; j++) {
                    MAT_VAL(i, j) = operation(MAT_VAL(i, j), other->_get(i, j));
                }
            }
        }
        // void applyComponentBinaryOp(const std::shared_ptr<Matrix>& other, float (*operation)(float, float));

        template<typename Lambda>
        void applyComponentBinaryOpInto(const std::shared_ptr<Matrix>& other, Lambda &&operation, const std::shared_ptr<Matrix> &dest) {
            if (!sameSize(other)) throw MatrixSizeMismatchException();
            if (!sameSize(dest)) throw MatrixSizeMismatchException();

            for (size_t i = 0; i < m_height; i++) {
                for (size_t j = 0; j < m_width; j++) {
                    dest->set(i, j, operation(MAT_VAL(i, j), other->_get(i, j)));
                }
            }
        }

        // void applyComponentBinaryOp(const std::shared_ptr<Matrix>& other, float (*operation)(float, float));


        template<typename Lambda>
        std::shared_ptr<Matrix> returnComponentBinaryOp(const std::shared_ptr<Matrix>& other, Lambda &&operation);

        template<typename Lambda>
        void mapInplace(Lambda &&map);

        template<typename Lambda>
        void mapFrom(const std::shared_ptr<Matrix>& src, Lambda &&map) {
            if (!sameSize(src)) throw MatrixSizeMismatchException();
            std::transform(src->data, src->data + (m_height * m_width), data, map);
        }

        // functions to initialize or overwrite large chunks or entire matrix, ignoring matrix structure
        void overwrite(float overwriteValue);

        void initRandom(std::mt19937 rng, std::uniform_real_distribution<> distribution);

        /**
         * Copies as many values from vec into data array, ignores matrix structure
         * @param vec input vector
         */
        void initFromStdVector(std::vector<float> &vec);

        void initFromUntil(const std::shared_ptr<Matrix>& src, size_t count);

        std::shared_ptr<Matrix> copy();

        /**
         * Task specific utility function. Adds an extra element to column vector with a value of 1 to the bottom
         * @return column vector extended by 1
         */
        std::shared_ptr<Matrix> copyAndAddBiasTerm();

        void print();

        std::string asString(int indent = 0, int width = 8, int precision = 3);

    private:

        // Internal get method, doesnt check values, should always be used even for internal access, because it considers m_isTransposed status
        [[nodiscard]] float _get(size_t i, size_t j) const;

    };


}

#endif //NEURALNET_MATRIX_H
