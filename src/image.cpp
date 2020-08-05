#include <fstream>

#include "image.hpp"

#include "data_load_failure.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

Image::Image(const char* fileName)
{
    int n = 0;
    unsigned char* data = stbi_load(fileName, &width_, &height_, &n, 0);
    if(!data)
        throw data_load_failure(fileName);

    if(n != 3)
        throw data_load_failure(fileName, " Please make sure that the image has 3 components.");

    const int len = 3*width_*height_;
    pixels_ = std::make_unique<unsigned char[]>(len);
    memcpy(pixels_.get(), data, len * sizeof(unsigned char)); // Copying is easier than using custom deleter for unique_ptr :) 
    stbi_image_free(data);
}

Image::Image(const Image& other)
{
    width_ = other.width_;
    height_ = other.height_;

    int len = 3*width_*height_;
    pixels_ = std::make_unique<unsigned char[]>(len);

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
        pixels_ = std::make_unique<unsigned char[]>(len);

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

unsigned char Image::operator[](unsigned int index) const
{
    return pixels_[index];
}

unsigned char& Image::operator[](unsigned int index)
{
    return pixels_[index];
}
