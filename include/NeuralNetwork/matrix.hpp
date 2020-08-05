#pragma once

#include <chrono>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <type_traits>

template<typename T>
class Matrix
{
public:
    typedef T* iterator;
    typedef const T* const_iterator;

    Matrix(unsigned int rows, unsigned int columns);
    explicit Matrix(): Matrix(1, 1) {}
    Matrix(const T* data, unsigned int rows, unsigned int columns);
    Matrix(const Matrix& o): Matrix(o.data_.get(), o.rows_, o.columns_) {}
    Matrix(Matrix&& o);

    const_iterator cbegin() const { return data_.get(); }
    const_iterator cend() const { return data_ .get() + len_; }
    
    iterator begin() { return data_.get(); }
    iterator end() { return data_.get() + len_; }

    unsigned int getRows() const;
    unsigned int getColumns() const;
    const T* getData() const;
    T get(unsigned int i, unsigned int j) const;

    // Sets all values to zero
    void zero();

    // Sets values to random in range [min, max)
    void randomize(T min, T max);

    // Apply function to every matrix entry
    Matrix map(std::function<T(T)> const& f) const;
    T sum() const;

    // Element-wise multiplication
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

    const T& operator[](int index) const { return data_[index]; }
    T& operator[](int index) { return data_[index]; }

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
    std::unique_ptr<T[]> data_;

    unsigned int at(unsigned int i, unsigned int j) const
    {
        return i*columns_ + j;
    }
};

template<typename T>
Matrix<T>::Matrix(unsigned int rows, unsigned int columns): rows_(rows), columns_(columns)
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator_ = std::mt19937(seed);

    len_ = rows*columns;
    data_ = std::make_unique<T[]>(len_);

    for(unsigned int i = 0; i < len_; ++i)
    {
        data_[i] = 1;
    }
}

template<typename T>
Matrix<T>::Matrix(const T* data, unsigned int rows, unsigned int columns): rows_(rows), columns_(columns)
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator_ = std::mt19937(seed);

    len_ = rows*columns;
    data_ = std::make_unique<T[]>(len_);
    for(unsigned int i = 0; i < len_; ++i)
    {
        data_[i] = data[i];
    }
}

template<typename T>
Matrix<T>::Matrix(Matrix&& o)
{
    data_ = std::move(o.data_);
    rows_ = o.rows_;
    columns_ = o.columns_;
    len_ = o.len_;
    generator_ = std::move(o.generator_);
}

template<typename T>
unsigned int Matrix<T>::getRows() const
{
    return rows_;
}

template<typename T>
unsigned int Matrix<T>::getColumns() const
{
    return columns_;
}

template<typename T>
const T* Matrix<T>::getData() const
{
    return data_.get();
}

template<typename T>
T Matrix<T>::get(unsigned int i, unsigned int j) const
{
    if(i < 0 || i >= rows_ || j < 0 || j >= columns_)
    {
        std::stringstream ss;
        ss << "ERROR: Cannot access matrix entry indexed " << i << j << "!\n";
        throw std::runtime_error(ss.str());
    }
    return data_[at(i, j)];
}

template<typename T>
void Matrix<T>::zero()
{
    for(unsigned int i = 0; i < len_; ++i)
    {
        data_[i] = 0;
    }
}

template<typename T>
void Matrix<T>::randomize(T min, T max)
{
    if constexpr(std::is_integral<T>::value)
    {
        std::uniform_int_distribution<T> distribution(min, max);
        for(unsigned int i = 0; i < len_; ++i)
        {
            data_[i] = distribution(generator_);
        }
    }
    else if constexpr(std::is_floating_point<T>::value)
    {
        std::uniform_real_distribution<T> distribution(min, max);
        for(unsigned int i = 0; i < len_; ++i)
        {
            data_[i] = distribution(generator_);
        }
    }
}

template<typename T>
Matrix<T> Matrix<T>::map(std::function<T(T)> const& f) const
{
    Matrix result(rows_, columns_);
    for(unsigned int i = 0; i < len_; ++i)
    {
        result.data_[i] = f(data_[i]);
        if(std::isinf(result.data_[i]) || std::isnan(result.data_[i]))
        {
            result.data_[i] = 0;
        }
    }
    return result;
}

template<typename T>
T Matrix<T>::sum() const
{
    T sum = 0;
    for(unsigned int i = 0; i < len_; ++i)
    {
        sum += data_[i];
    }
    return sum;
}

template<typename T>
Matrix<T> Matrix<T>::hadamard(const Matrix<T>& o) const
{
    if(o.rows_ != rows_ || o.columns_ != columns_)
    {
        throw std::runtime_error("ERROR: Cannot perform hadamard product on matrices with different sizes!\n");
    }
    Matrix result(rows_, columns_);
    for(unsigned int i = 0; i < len_; ++i)
    {
        result.data_[i] = data_[i]*o.data_[i];
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::transpose(const Matrix<T>& m)
{
    // inverted on purpose
    Matrix result(m.columns_, m.rows_);
    for(unsigned int i = 0; i < m.columns_; ++i)
    {
        for(unsigned int j = 0; j < m.rows_; ++j)
        {
            result.data_[result.at(i, j)] = m.data_[m.at(j, i)]; // here also
        }
    }
    return result;
}

template<typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix& o)
{
    if(&o != this)
    {
        if(len_ != o.len_)
            data_ = std::make_unique<T[]>(len_);
        
        rows_ = o.rows_;
        columns_ = o.columns_;
        len_ = o.len_;

        for(unsigned int i = 0; i < len_; ++i)
        {
            data_[i] = o.data_[i];
        }
    }
    
    return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::operator=(Matrix&& o)
{
    if(&o != this)
    {
        data_ = std::move(o.data_);
        rows_ = o.rows_;
        columns_ = o.columns_;
        len_ = o.len_;
        generator_ = std::move(o.generator_);
    }
    return *this;
}

template<typename T>
Matrix<T> Matrix<T>::operator+(const Matrix& o) const
{
    if(o.rows_ != rows_ || o.columns_ != columns_)
    {
        throw std::runtime_error("ERROR: Cannot perform addition of matrices with different sizes!\n");
    }
    Matrix result(rows_, columns_);
    for(unsigned int i = 0; i < len_; ++i)
    {
        result.data_[i] = data_[i] + o.data_[i];
    }
    return result;
}

template<typename T>
Matrix<T>& Matrix<T>::operator+=(const Matrix& o)
{
    if(o.rows_ != rows_ || o.columns_ != columns_)
    {
        throw std::runtime_error("ERROR: Cannot perform addition of matrices with different sizes!\n");
    }
    for(unsigned int i = 0; i < len_; ++i)
    {
        data_[i] += o.data_[i];
    }
    return *this;
}

template<typename T>
Matrix<T> Matrix<T>::operator-(const Matrix& o) const
{
    if(o.rows_ != rows_ || o.columns_ != columns_)
    {
        throw std::runtime_error("ERROR: Cannot perform subtraction of matrices with different sizes!\n");
    }
    Matrix result(rows_, columns_);
    for(unsigned int i = 0; i < len_; ++i)
    {
        result.data_[i] = data_[i] - o.data_[i];
    }
    return result;
}

template<typename T>
Matrix<T>& Matrix<T>::operator-=(const Matrix& o)
{
    if(o.rows_ != rows_ || o.columns_ != columns_)
    {
        throw std::runtime_error("ERROR: Cannot perform addition of matrices with different sizes!\n");
    }
    for(unsigned int i = 0; i < len_; ++i)
    {
        data_[i] -= o.data_[i];
    }
    return *this;
}

template<typename T>
Matrix<T> Matrix<T>::operator*(T f) const
{
    Matrix result(rows_, columns_);
    for(unsigned int i = 0; i < len_; ++i)
    {
        result.data_[i] = data_[i]*f;
    }
    return result;
}

template<typename T>
Matrix<T>& Matrix<T>::operator*=(T f)
{
    for(unsigned int i = 0; i < len_; ++i)
    {
        data_[i] *= f;
    }
    return *this;
}

template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix& o) const
{
    if(columns_ != o.rows_)
    {
        throw std::runtime_error("ERROR: Inappropriate sizes of matrices to perform multiplication!\n");
    }
    Matrix result(rows_, o.columns_);
    T s;
    for(unsigned int i = 0; i < rows_; ++i)
    {
        for(unsigned int j = 0; j < o.columns_; ++j)
        {
            s = 0;
            for(unsigned int r = 0; r < columns_; ++r)
            {
                s += data_[at(i, r)]*o.data_[o.at(r, j)];
            }
            result.data_[result.at(i, j)] = s;
        }
    }
    return result;
}
