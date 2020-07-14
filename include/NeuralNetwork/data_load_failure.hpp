#pragma once

#include <exception>
#include <sstream>

class data_load_failure : public std::exception
{
public:
    data_load_failure(const char* filename, const char* additional = "")
    {
        std::stringstream ss;
        ss << "Could not load data from file named " << filename << "!" << additional;
        message_ = ss.str();
    }

    const char* what() const throw()
    {
    	return message_.c_str();
    }
private:
    std::string message_;
};
