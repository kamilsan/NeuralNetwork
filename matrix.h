#pragma once

#include <iostream>
#include <random>
#include <chrono>
#include <type_traits>
#include <functional>

template<typename T>
class Matrix
{
public:
    Matrix(unsigned int rows, unsigned int columns);
    explicit Matrix(): Matrix(1, 1) {}
    Matrix(const T* data, unsigned int rows, unsigned int columns);
    Matrix(const Matrix& o): Matrix(o.data_, o.rows_, o.columns_) {}
    Matrix(Matrix&& o);

    ~Matrix();

    unsigned int getRows() const;
    unsigned int getColumns() const;
    const T* getData() const;
    T get(unsigned int i, unsigned int j) const;

    void zero();
    void randomize(T min, T max);
    Matrix map(std::function<T(T)> const& f) const;
    T sum() const;
    Matrix hadamard(const Matrix& o) const;
    Matrix static transpose(const Matrix& m);

    Matrix& operator=(const Matrix& o);
    Matrix& operator=(Matrix&& o);
    Matrix operator+(const Matrix& o) const;
    Matrix& operator+=(const Matrix& o);
    Matrix operator-(const Matrix& o) const;
    Matrix& operator-=(const Matrix& o);
    Matrix operator*(T f) const;
    Matrix& operator*=(T f);

    Matrix operator*(const Matrix& o) const;

    friend Matrix operator*(T f, const Matrix& m)
    {
        return m*f;
    }
    friend std::ostream& operator<<(std::ostream& os, const Matrix& m)
    {
        for(unsigned int i = 0, k = 0; i < m.rows_; ++i)
        {
            os << "[";
            for(unsigned int j = 0; j < m.columns_; ++j, ++k)
            {
                os << m.data_[k];
                if(j + 1 < m.columns_) os << "\t";
            }
            os << "]";
            if(i + 1 < m.rows_) os << "\n";
        }
        return os;
    }
private:
    unsigned int rows_;
    unsigned int columns_;
    unsigned int len_;
    std::mt19937 generator_;
    T* data_;

    unsigned int at(unsigned int i, unsigned int j) const
    {
        return i*columns_ + j;
    }
};

 #include "matrix.tpp"