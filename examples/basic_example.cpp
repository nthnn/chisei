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

// Include Chisei library headers for activation functions and the neural network class
#include <chisei/activation_functions.hpp>
#include <chisei/neural_network.hpp>

int main() {
    // Define the input data for the XNOR gate (truth table)
    std::vector<std::vector<double>> inputs = {
        {0, 0}, // Input: 0 AND 0
        {0, 1}, // Input: 0 AND 1
        {1, 0}, // Input: 1 AND 0
        {1, 1}  // Input: 1 AND 1
    };

    // Define the expected target outputs for the XNOR gate
    std::vector<std::vector<double>> targets = {
        {1}, // Output: 1 (0 XNOR 0)
        {0}, // Output: 0 (0 XNOR 1)
        {0}, // Output: 0 (1 XNOR 0)
        {1}  // Output: 1 (1 XNOR 1)
    };

    // Create a neural network with 2 input neurons, 1 hidden layer of 4 neurons, and 1 output neuron.
    // Use the sigmoid activation function and its derivative for the network.
    chisei::NeuralNetwork xnor(
        {2, 4, 1}, // Network architecture: 2 inputs -> 4 hidden neurons -> 1 output
        chisei::ActivationFunctions::sigmoid_activation, // Sigmoid activation function
        chisei::ActivationFunctions::sigmoid_derivative  // Sigmoid derivative for backpropagation
    );

    // Train the neural network with the input and target data.
    // Use a high learning rate (6) and a default number of epochs.
    xnor.train(inputs, targets, 6);

    // Test the trained network by making predictions for each input
    for (const auto& input : inputs) {
        auto prediction = xnor.predict(input); // Predict the output for the given input
        std::cout << "Input: ["
            << input[0] << ", " << input[1]  // Display the input values
            << "]\tPrediction: "
            << (prediction[0] >= 0.5 ? "1.0" : "0.0") // Output a binary prediction (threshold 0.5)
            << "\tRaw: " << prediction[0]   // Display the raw output value
            << std::endl;
    }

    // Save the trained model to a file for later use
    xnor.save_model("data/xnor_model.chisei");

    // Load the saved model from the file
    chisei::NeuralNetwork loaded_model = chisei::NeuralNetwork::loadFromModel("data/xnor_model.chisei");

    // Compute the accuracy of the loaded model on the training data
    double accuracy = loaded_model.compute_accuracy(inputs, targets);
    std::cout << "Network Accuracy: " << accuracy * 100 << "%" << std::endl; // Display the accuracy in percentage

    // Test the loaded model with the same input data to verify consistency
    for (const auto& input : inputs) {
        auto prediction = loaded_model.predict(input); // Predict using the loaded model
        std::cout << "Input: ["
            << input[0] << ", " << input[1]  // Display the input values
            << "]\tPrediction: "
            << (prediction[0] >= 0.5 ? "1.0" : "0.0") // Output a binary prediction (threshold 0.5)
            << "\tRaw: " << prediction[0]   // Display the raw output value
            << std::endl;
    }

    return 0; // Exit the program
}
