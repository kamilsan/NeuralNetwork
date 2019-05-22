#include "matrix.h"

template<typename T>
Matrix<T>::Matrix(int rows, int columns): rows_(rows), columns_(columns)
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

template<typename T>
Matrix<T>::Matrix(const T* data, int rows, int columns): rows_(rows), columns_(columns)
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

template<typename T>
Matrix<T>::Matrix(Matrix&& o)
{
    data_ = o.data_;
    rows_ = o.rows_;
    columns_ = o.columns_;
    len_ = o.len_;
    generator_ = std::move(o.generator_);
    o.data_ = nullptr;
}

template<typename T>
Matrix<T>::~Matrix()
{
    delete[] data_;
}

template<typename T>
int Matrix<T>::getRows() const
{
    return rows_;
}

template<typename T>
int Matrix<T>::getColumns() const
{
    return columns_;
}

template<typename T>
const T* Matrix<T>::getData() const
{
    return data_;
}

template<typename T>
T Matrix<T>::get(int i, int j) const
{
    if(i < 0 || i >= rows_ || j < 0 || j >= columns_)
    {
        std::cout << "ERROR: Cannot access matrix entry indexed " << i << j << "!\n";
        return 0;
    }
    return data_[at(i, j)];
}

template<typename T>
void Matrix<T>::zero()
{
    for(int i = 0; i < len_; ++i)
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

template<typename T>
Matrix<T> Matrix<T>::map(T (*f)(T))
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

template<typename T>
T Matrix<T>::sum() const
{
    T sum = 0;
    for(int i = 0; i < len_; ++i)
    {
        sum += data_[i];
    }
    return sum;
}

template<typename T>
Matrix<T> Matrix<T>::hadamard(const Matrix<T>& o)
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

template<typename T>
Matrix<T> Matrix<T>::transpose(const Matrix<T>& m)
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

template<typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix& o)
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

template<typename T>
Matrix<T>& Matrix<T>::operator=(Matrix&& o)
{
    if(&o != this)
    {
        if(data_) delete[] data_;
        data_ = o.data_;
        rows_ = o.rows_;
        columns_ = o.columns_;
        len_ = o.len_;
        generator_ = std::move(o.generator_);
        o.data_ = nullptr;
    }
    return *this;
}

template<typename T>
Matrix<T> Matrix<T>::operator+(const Matrix& o) const
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

template<typename T>
Matrix<T>& Matrix<T>::operator+=(const Matrix& o)
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

template<typename T>
Matrix<T> Matrix<T>::operator-(const Matrix& o) const
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

template<typename T>
Matrix<T>& Matrix<T>::operator-=(const Matrix& o)
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

template<typename T>
Matrix<T> Matrix<T>::operator*(T f) const
{
    Matrix result(rows_, columns_);
    for(int i = 0; i < len_; ++i)
    {
        result.data_[i] = data_[i]*f;
    }
    return result;
}

template<typename T>
Matrix<T>& Matrix<T>::operator*=(T f)
{
    for(int i = 0; i < len_; ++i)
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