#pragma once

#include <memory>

class Image
{
public:
    Image(const char* fileName);
    Image(const Image& other);
    Image(Image&& other);

    int getWidth() const;
    int getHeight() const;

    Image& operator=(const Image& other);
    Image& operator=(Image&& other);
    char operator[](unsigned int index) const;
    char& operator[](unsigned int index);
private:
    int width_;
    int height_;
    std::unique_ptr<char[]> pixels_;
};