#pragma once

class Image
{
public:
    Image(const char* fileName);
    Image(const Image& other);
    ~Image();

    int getWidth() const;
    int getHeight() const;
    const char* getPixels() const;

    Image& operator=(const Image& o);
private:
    int width_;
    int height_;
    char* pixels_;
};