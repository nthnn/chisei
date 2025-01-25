/*
 * 
 * Copyright 2025 Nathanne Isip
 * 
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted provided
 * that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the
 *    above copyright notice, this list of conditions
 *    and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce
 *    the above copyright notice, this list of conditions
 *    and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 * 
 */

#include <chisei/idx_loader.hpp>

namespace chisei {

NeuralNetwork IDXLoader::fromMNIST(
    const std::string& images_file,
    const std::string& labels_file,
    double learning_rate,
    int epoch
) {
    std::ifstream images(images_file, std::ios::binary);
    std::ifstream labels(labels_file, std::ios::binary);

    if(!images || !labels) 
        throw std::bad_exception("Failed to open MNIST files");

    uint32_t image_magic = readUint32(images);
    uint32_t num_images = readUint32(images);
    uint32_t rows = readUint32(images);
    uint32_t cols = readUint32(images);

    uint32_t label_magic = readUint32(labels);
    readUint32(labels);

    if(image_magic != 0x00000803 || label_magic != 0x00000801)
        throw std::bad_exception("Invalid MNIST file format");

    size_t input_size = rows * cols;
    size_t output_size = 10;

    std::vector<size_t> layer_sizes = {
        input_size,
        256,
        128,
        output_size
    };

    NeuralNetwork network(
        layer_sizes,
        ActivationFunctions::sigmoid_activation,
        ActivationFunctions::sigmoid_derivative
    );

    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> targets;

    uint32_t max_samples = std::min(num_images, 5000u);
    for(uint32_t i = 0; i < max_samples; ++i) {
        std::vector<double> input(input_size);

        for(size_t j = 0; j < input_size; ++j) {
            uint8_t pixel;

            images.read(reinterpret_cast<char*>(&pixel), 1);
            input[j] = static_cast<double>(pixel) / 255.0;
        }

        uint8_t label;
        labels.read(reinterpret_cast<char*>(&label), 1);

        std::vector<double> target(output_size, 0.0);
        target[label] = 1.0;

        inputs.push_back(input);
        targets.push_back(target);
    }

    network.train(inputs, targets, learning_rate, epoch);
    return network;
}

uint32_t IDXLoader::readUint32(std::ifstream& file) {
    uint32_t value;
    file.read(reinterpret_cast<char*>(&value), sizeof(value));

    return (value >> 24)            |
        ((value << 8) & 0x00FF0000) | 
        ((value >> 8) & 0x0000FF00) | 
        (value << 24);
}

}
