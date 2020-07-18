#include <fstream>

#include "image.hpp"

#include "data_load_failure.hpp"

Image::Image(const char* fileName)
{
    std::ifstream file;
    file.open(fileName, std::ios::binary);
    if(!file.is_open()) throw data_load_failure(fileName);

    char h[4];
    file.read(h, 3);
    if(h[0] != 'P' || h[1] != '6') 
    { 
        throw data_load_failure(fileName, " Please make sure that the file has a correct format."); 
    }

    while(file.get() == '#')
    {
        while(file.get() != '\n');
    }
    file.unget();

    int max;
    file >> width_ >> height_ >> max;
    file.get();

    int len = 3*width_*height_;
    pixels_ = std::make_unique<char[]>(len);
    file.read(pixels_.get(), len);

    file.close();
}

Image::Image(const Image& other)
{
    width_ = other.width_;
    height_ = other.height_;

    int len = 3*width_*height_;
    pixels_ = std::make_unique<char[]>(len);

    for(int i = 0; i < len; ++i)
    {
        pixels_[i] = other.pixels_[i];
    }
}

Image::Image(Image&& other)
{
    width_ = other.width_;
    height_ = other.height_;
    pixels_ = std::move(other.pixels_);
}

int Image::getWidth() const
{
    return width_;
}

int Image::getHeight() const
{
    return height_;
}

Image& Image::operator=(const Image& other)
{
    if(&other != this)
    {
        width_ = other.width_;
        height_ = other.height_;

        int len = 3*width_*height_;
        pixels_ = std::make_unique<char[]>(len);

        for(int i = 0; i < len; ++i)
        {
            pixels_[i] = other.pixels_[i];
        }
    }

    return *this;
}

Image& Image::operator=(Image&& other)
{
    if(&other != this)
    {
        width_ = other.width_;
        height_ = other.height_;
        pixels_ = std::move(other.pixels_);
    }

    return *this;
}

char Image::operator[](unsigned int index) const
{
    return pixels_[index];
}

char& Image::operator[](unsigned int index)
{
    return pixels_[index];
}
