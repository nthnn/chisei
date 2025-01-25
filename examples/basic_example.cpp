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

#include <iostream>
#include <vector>

#include <chisei/activation_functions.hpp>
#include <chisei/neural_network.hpp>

int main() {
    std::vector<std::vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> targets = {{1}, {0}, {0}, {1}};

    chisei::NeuralNetwork xnor(
        {2, 4, 1},
        chisei::ActivationFunctions::sigmoid_activation,
        chisei::ActivationFunctions::sigmoid_derivative
    );
    xnor.train(inputs, targets, 6);

    for(const auto& input : inputs) {
        auto prediction = xnor.predict(input);
        std::cout << "Input: ["
            << input[0] << ", " << input[1]
            << "]\tPrediction: "
            << (prediction[0] >= 0.5 ? "1.0" : "0.0")
            << "\tRaw: " << prediction[0]
            << std::endl;
    }
    xnor.save_model("data/xnor_model.chisei");

    chisei::NeuralNetwork loaded_model = chisei::NeuralNetwork::loadFromModel("data/xnor_model.chisei");

    double accuracy = loaded_model.compute_accuracy(inputs, targets);
    std::cout << "Network Accuracy: " << accuracy * 100 << "%" << std::endl;

    for(const auto& input : inputs) {
        auto prediction = loaded_model.predict(input);
        std::cout << "Input: ["
            << input[0] << ", " << input[1]
            << "]\tPrediction: "
            << (prediction[0] >= 0.5 ? "1.0" : "0.0")
            << "\tRaw: " << prediction[0]
            << std::endl;
    }

    return 0;
}
