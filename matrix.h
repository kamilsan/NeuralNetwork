#pragma once

#include <iostream>

template<typename T>
class Matrix
{
public:
    int rows;//TODO: getters&setters
    int columns;

    Matrix(): rows(1), columns(1)
    {
        data_ = new T[rows*columns];
        for(int i = 0; i < rows; ++i)
        {
            for(int j = 0; j < columns; ++j)
            data_[at(i, j)] = 1;
        }
    }

    Matrix(int rows_, int columns_): rows(rows_), columns(columns_)
    {
        data_ = new T[rows*columns];
        for(int i = 0; i < rows; ++i)
        {
            for(int j = 0; j < columns; ++j)
            data_[at(i, j)] = 1;
        }
    }

    Matrix(const T* data, int rows_, int columns_): rows(rows_), columns(columns_)
    {
        int k = 0;
        data_ = new T[rows*columns];
        for(int i = 0; i < rows; ++i)
        {
            for(int j = 0; j < columns; ++j)
            {
            data_[at(i, j)] = data[k++];
            }
        }
    }

    Matrix(const Matrix& o): rows(o.rows), columns(o.columns)
    {
        data_ = new T[rows*columns];
        for(int i = 0; i < rows; ++i)
        {
            for(int j = 0; j < columns; ++j)
            data_[at(i, j)] = o.data_[at(i, j)];
        }
    }

    ~Matrix()
    {
        delete[] data_;
    }


    T get(int i, int j) const
    {
        if(i < 0 || i >= rows || j < 0 || j >= columns)
        {
            std::cout << "ERROR: Cannot access matrix entry indexed " << i << j << "!\n";
            return 0;
        }
        return data_[at(i, j)];
    }

    Matrix hadamard(const Matrix& o)
    {
        if(o.rows != rows || o.columns != columns)
        {
            std::cout << "ERROR: Cannot perform hadamard product on matrices with different sizes!\n";
            return *this;
        }
        Matrix result(rows, columns);
        for(int i = 0; i < rows; ++i)
        {
            for(int j = 0; j < columns; ++j)
            {
                result.data_[at(i, j)] = data_[at(i, j)]*o.data_[at(i, j)];
            }
        }
        return result;
    }

    Matrix static transpose(const Matrix& m)
    {
        //inverted on purpose
        Matrix result(m.columns, m.rows);
        for(int i = 0; i < m.columns; ++i)
        {
            for(int j = 0; j < m.rows; ++j)
            {
                result.data_[result.at(i, j)] = m.data_[m.at(j, i)];
            }
        }
        return result;
    }

    Matrix& operator=(const Matrix& o)
    {
        if(this == &o)
            return *this;

        delete[] data_;

        rows = o.rows;
        columns = o.columns;
        data_ = new T[rows*columns];
        for(int i = 0; i < rows; ++i)
        {
            for(int j = 0; j < columns; ++j)
            data_[at(i, j)] = o.data_[at(i, j)];
        }

        return *this;
    }

    Matrix operator+(const Matrix& o) const
    {
        if(o.rows != rows || o.columns != columns)
        {
            std::cout << "ERROR: Cannot perform addition of matrices with different sizes!\n";
            return *this;
        }
        Matrix result(rows, columns);
        for(int i = 0; i < o.rows; ++i)
        {
            for(int j = 0; j < o.columns; ++j)
            {
            result.data_[at(i, j)] = data_[at(i, j)] + o.data_[at(i, j)];
            }
        }
        return result;
    }

    Matrix& operator+=(const Matrix& o)
    {
        if(o.rows != rows || o.columns != columns)
        {
            std::cout << "ERROR: Cannot perform addition of matrices with different sizes!\n";
            return *this;
        }
        for(int i = 0; i < o.rows; ++i)
        {
            for(int j = 0; j < o.columns; ++j)
            {
            data_[at(i, j)] += o.data_[at(i, j)];
            }
        }
        return *this;
    }

    Matrix operator-(const Matrix& o) const
    {
        if(o.rows != rows || o.columns != columns)
        {
            std::cout << "ERROR: Cannot perform subtraction of matrices with different sizes!\n";
            return *this;
        }
        Matrix result(rows, columns);
        for(int i = 0; i < rows; ++i)
        {
            for(int j = 0; j < columns; ++j)
            {
            result.data_[at(i, j)] = data_[at(i, j)] - o.data_[at(i, j)];
            }
        }
        return result;
    }

    Matrix& operator-=(const Matrix& o)
    {
        if(o.rows != rows || o.columns != columns)
        {
            std::cout << "ERROR: Cannot perform addition of matrices with different sizes!\n";
            return *this;
        }
        for(int i = 0; i < o.rows; ++i)
        {
            for(int j = 0; j < o.columns; ++j)
            {
            data_[at(i, j)] -= o.data_[at(i, j)];
            }
        }
        return *this;
    }

    Matrix operator*(T f) const
    {
        Matrix result(rows, columns);
        for(int i = 0; i < rows; ++i)
        {
            for(int j = 0; j < columns; ++j)
            {
            result.data_[at(i, j)] = data_[at(i, j)]*f;
            }
        }
        return result;
    }

    Matrix& operator*=(T f)
    {
        for(int i = 0; i < rows; ++i)
        {
            for(int j = 0; j < columns; ++j)
            {
            data_[at(i, j)] *= f;
            }
        }
        return *this;
    }

    Matrix operator*(const Matrix& o) const
    {
        if(columns != o.rows)
        {
            std::cout << "ERROR: Inappropriate sizes of matrices to perform multiplication!\n";
            return *this;
        }
        Matrix result(rows, o.columns);
        T s;
        for(int i = 0; i < rows; ++i)
        {
            for(int j = 0; j < o.columns; ++j)
            {
            s = 0;
            for(int r = 0; r < columns; ++r)
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
        for(int i = 0; i < m.rows; ++i)
        {
            os << "[";
            for(int j = 0; j < m.columns; ++j)
            {
            os << m.data_[i];
            if(j + 1 < m.columns) os << "\t";
            }
            os << "]";
            if(i + 1 < m.rows) os << "\n";
        }
        return os;
    }
private:
    T *data_;

    int at(int i, int j) const
    {
        return i*columns + j;
    }
};