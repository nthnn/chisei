#ifndef CHISEI_MODEL_LOADER_EXCEPTION_HPP
#define CHISEI_MODEL_LOADER_EXCEPTION_HPP

#include <exception>
#include <string>

class ModelLoaderException final : public std::exception {
private:
    std::string message;

public:
    ModelLoaderException(std::string _message) :
        message(_message) { }

    char* what() {
        return const_cast<char*>(this->message.c_str());
    }
};

#endif
