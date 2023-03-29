//
//

#ifndef NEURALNET_MATRIX_H
#define NEURALNET_MATRIX_H


#include <cstddef>
#include <memory>

namespace NNet {

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

        std::shared_ptr<Matrix> multiply(Matrix &other);

        size_t width() const { return m_width; };
        size_t height() const { return m_height; };

        void print();


    };

}

#endif //NEURALNET_MATRIX_H
