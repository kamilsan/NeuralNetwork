#pragma once

#include <iostream>
#include <random>
#include <chrono>
#include <type_traits>

template<typename T>
class Matrix
{
public:
    Matrix(int rows, int columns): rows_(rows), columns_(columns)
    {
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        generator_ = std::mt19937(seed);

        len_ = rows*columns;
        data_ = new T[len_];

        for(int i = 0; i < len_; ++i)
        {
            data_[i] = 1;
        }
    }

    explicit Matrix(): Matrix(1, 1) {}

    Matrix(const T* data, int rows, int columns): rows_(rows), columns_(columns)
    {
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        generator_ = std::mt19937(seed);

        len_ = rows*columns;
        data_ = new T[len_];
        for(int i = 0; i < len_; ++i)
        {
            data_[i] = data[i];
        }
    }

    Matrix(const Matrix& o): Matrix(o.data_, o.rows_, o.columns_) {}

    ~Matrix()
    {
        delete[] data_;
    }

    int getRows() const
    {
        return rows_;
    }

    int getColumns() const
    {
        return columns_;
    }

    const T* getData() const
    {
        return data_;
    }

    T get(int i, int j) const
    {
        if(i < 0 || i >= rows_ || j < 0 || j >= columns_)
        {
            std::cout << "ERROR: Cannot access matrix entry indexed " << i << j << "!\n";
            return 0;
        }
        return data_[at(i, j)];
    }

    void zero()
    {
        for(int i = 0; i < len_; ++i)
        {
            data_[i] = 0;
        }
    }

    void randomize(T min, T max)
    {
        if constexpr(std::is_integral<T>::value)
        {
            std::uniform_int_distribution<T> distribution(min, max);
            for(int i = 0; i < len_; ++i)
            {
                data_[i] = distribution(generator_);
            }
        }
        else if constexpr(std::is_floating_point<T>::value)
        {
            std::uniform_real_distribution<T> distribution(min, max);
            for(int i = 0; i < len_; ++i)
            {
                data_[i] = distribution(generator_);
            }
        }
    }

    Matrix map(T (*f)(T))
    {
        Matrix result(rows_, columns_);
        for(int i = 0; i < len_; ++i)
        {
            result.data_[i] = f(data_[i]);
            if(std::isinf(result.data_[i]) || std::isnan(result.data_[i]))
            {
                result.data_[i] = 0;
            }
        }
        return result;
    }

    float sum() const
    {
        float sum = 0;
        for(int i = 0; i < len_; ++i)
        {
            sum += data_[i];
        }
        return sum;
    }

    Matrix hadamard(const Matrix& o)
    {
        if(o.rows_ != rows_ || o.columns_ != columns_)
        {
            std::cout << "ERROR: Cannot perform hadamard product on matrices with different sizes!\n";
            return *this;
        }
        Matrix result(rows_, columns_);
        for(int i = 0; i < len_; ++i)
        {
            result.data_[i] = data_[i]*o.data_[i];
        }
        return result;
    }

    Matrix static transpose(const Matrix& m)
    {
        //inverted on purpose
        Matrix result(m.columns_, m.rows_);
        for(int i = 0; i < m.columns_; ++i)
        {
            for(int j = 0; j < m.rows_; ++j)
            {
                result.data_[result.at(i, j)] = m.data_[m.at(j, i)];
            }
        }
        return result;
    }

    Matrix& operator=(const Matrix& o)
    {
        float* old = data_;
        rows_ = o.rows_;
        columns_ = o.columns_;
        len_ = o.len_;
        data_ = new T[len_];

        for(int i = 0; i < len_; ++i)
        {
            data_[i] = o.data_[i];
        }

        delete[] old;
        
        return *this;
    }

    Matrix operator+(const Matrix& o) const
    {
        if(o.rows_ != rows_ || o.columns_ != columns_)
        {
            std::cout << "ERROR: Cannot perform addition of matrices with different sizes!\n";
            return *this;
        }
        Matrix result(rows_, columns_);
        for(int i = 0; i < len_; ++i)
        {
            result.data_[i] = data_[i] + o.data_[i];
        }
        return result;
    }

    Matrix& operator+=(const Matrix& o)
    {
        if(o.rows_ != rows_ || o.columns_ != columns_)
        {
            std::cout << "ERROR: Cannot perform addition of matrices with different sizes!\n";
            return *this;
        }
        for(int i = 0; i < len_; ++i)
        {
            data_[i] += o.data_[i];
        }
        return *this;
    }

    Matrix operator-(const Matrix& o) const
    {
        if(o.rows_ != rows_ || o.columns_ != columns_)
        {
            std::cout << "ERROR: Cannot perform subtraction of matrices with different sizes!\n";
            return *this;
        }
        Matrix result(rows_, columns_);
        for(int i = 0; i < len_; ++i)
        {
            result.data_[i] = data_[i] - o.data_[i];
        }
        return result;
    }

    Matrix& operator-=(const Matrix& o)
    {
        if(o.rows_ != rows_ || o.columns_ != columns_)
        {
            std::cout << "ERROR: Cannot perform addition of matrices with different sizes!\n";
            return *this;
        }
        for(int i = 0; i < len_; ++i)
        {
            data_[i] -= o.data_[i];
        }
        return *this;
    }

    Matrix operator*(T f) const
    {
        Matrix result(rows_, columns_);
        for(int i = 0; i < len_; ++i)
        {
            result.data_[i] = data_[i]*f;
        }
        return result;
    }

    Matrix& operator*=(T f)
    {
        for(int i = 0; i < len_; ++i)
        {
            data_[i] *= f;
        }
        return *this;
    }

    Matrix operator*(const Matrix& o) const
    {
        if(columns_ != o.rows_)
        {
            std::cout << "ERROR: Inappropriate sizes of matrices to perform multiplication!\n";
            return *this;
        }
        Matrix result(rows_, o.columns_);
        T s;
        for(int i = 0; i < rows_; ++i)
        {
            for(int j = 0; j < o.columns_; ++j)
            {
                s = 0;
                for(int r = 0; r < columns_; ++r)
                {
                    s += data_[at(i, r)]*o.data_[o.at(r, j)];
                }
                result.data_[result.at(i, j)] = s;
            }
        }
        return result;
    }

    friend Matrix operator*(T f, const Matrix& m)
    {
        return m*f;
    }
    friend std::ostream& operator<<(std::ostream& os, const Matrix& m)
    {
        for(int i = 0, k = 0; i < m.rows_; ++i)
        {
            os << "[";
            for(int j = 0; j < m.columns_; ++j, ++k)
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
    int rows_;
    int columns_;
    int len_;
    std::mt19937 generator_;
    T* data_;

    int at(int i, int j) const
    {
        return i*columns_ + j;
    }
};