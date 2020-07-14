#include <fstream>

#include "image.hpp"

#include "data_load_failure.hpp"

Image::Image(const char* fileName)
{
    std::ifstream file;
    file.open(fileName, std::ios::binary);
    if(!file.is_open()) throw data_load_failure(fileName);

    char* h = new char[3];
    file.read(h, 3);
    if(h[0] != 'P' || h[1] != '6') 
    { 
        delete[] h; 
        throw data_load_failure(fileName, " Please make sure that the file has a correct format."); 
    }
    delete[] h;

    while(file.get() == '#')
    {
        while(file.get() != '\n');
    }
    file.unget();

    int max;
    file >> width_ >> height_ >> max;
    file.get();

    int len = 3*width_*height_;
    pixels_ = new char[len];
    file.read(pixels_, len);

    file.close();
}

Image::Image(const Image& other)
{
    width_ = other.width_;
    height_ = other.height_;

    int len = 3*width_*height_;
    pixels_ = new char[len];

    for(int i = 0; i < len; ++i)
    {
        pixels_[i] = other.pixels_[i];
    }
}

Image::~Image()
{
    delete[] pixels_;
}


int Image::getWidth() const
{
    return width_;
}

int Image::getHeight() const
{
    return height_;
}

const char* Image::getPixels() const
{
    return pixels_;
}

Image& Image::operator=(const Image& o)
{
    char* old = pixels_;
    width_ = o.width_;
    height_ = o.height_;

    int len = 3*width_*height_;
    pixels_ = new char[len];

    for(int i = 0; i < len; ++i)
    {
        pixels_[i] = o.pixels_[i];
    }

    delete[] old;
    
    return *this;
}
