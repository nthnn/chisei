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

/**
 * @file NeuralNetwork.hpp
 * @author [Nathanne Isip](https://github.com/nthnn)
 * @brief Header file for the NeuralNetwork class, providing functionality for building, training, 
 *        and using feedforward neural networks.
 */
#ifndef CHISEI_NEURAL_NETWORK_HPP
#define CHISEI_NEURAL_NETWORK_HPP

#include <algorithm>
#include <exception>
#include <fstream>
#include <functional>
#include <random>
#include <vector>

#include <chisei/activation_functions.hpp>
#include <chisei/cpu_feature_optimizer.hpp>

namespace chisei {

    /**
     * @class NeuralNetwork
     * @brief Represents a fully connected feedforward neural network.
     * 
     * This class provides methods for creating, training, and using neural networks. It supports:
     * - Customizable activation functions.
     * - Training via backpropagation with mean squared error (MSE) loss.
     * - Saving and loading models to/from files.
     */
    class NeuralNetwork {
    private:

        /**
         * @brief The size of each layer in the neural network.
         * 
         * For example, a network with layer sizes {3, 5, 2} has:
         * - An input layer with 3 neurons.
         * - A hidden layer with 5 neurons.
         * - An output layer with 2 neurons.
         */
        std::vector<size_t> layer_sizes;

        /**
         * @brief Weight matrices for each layer of the network.
         * 
         * Each weight matrix connects one layer to the next. The size of the matrix is
         * determined by the number of neurons in the current layer and the number of 
         * neurons in the next layer.
         */
        std::vector<std::vector<std::vector<double>>> weights;

        /**
         * @brief Bias vectors for each layer of the network.
         * 
         * Each bias vector corresponds to the neurons in a given layer, excluding the input layer.
         */
        std::vector<std::vector<double>> biases;

        /**
         * @brief The activation function used by the network.
         * 
         * The activation function is applied element-wise to the output of each layer.
         */
        std::function<double(double)> activation;

        /**
         * @brief The derivative of the activation function.
         * 
         * This is used during backpropagation to compute gradients.
         */
        std::function<double(double)> activation_derivative;

        /**
         * @brief Random number generator for initializing weights and biases.
         */
        std::random_device rd;

        /**
         * @brief Mersenne Twister random number generator.
         */
        std::mt19937 gen{rd()};

        /**
         * @brief Normal distribution for initializing weights.
         * 
         * Mean = 0, Standard Deviation = 0.1.
         */
        std::normal_distribution<> weight_dist{0, 0.1};

    public:
        /**
         * @brief Constructs a neural network with the specified layers and activation functions.
         * 
         * @param _layers A vector specifying the number of neurons in each layer.
         * @param _activation The activation function to use in the network.
         * @param _activation_derivative The derivative of the activation function.
         */
        NeuralNetwork(
            const std::vector<size_t>& _layers,
            std::function<double(double)> _activation,
            std::function<double(double)> _activation_derivative
        );

        /**
         * @brief Copy constructor.
         * 
         * Creates a deep copy of the given neural network.
         * 
         * @param other The neural network to copy.
         */
        NeuralNetwork(const NeuralNetwork& other);

        /**
         * @brief Destructor for the neural network.
         */
        ~NeuralNetwork();

        /**
         * @brief Move assignment operator.
         * 
         * Moves the contents of another neural network into this instance.
         * 
         * @param other The neural network to move from.
         * @return A reference to the current instance.
         */
        NeuralNetwork& operator=(NeuralNetwork&& other) noexcept;

        /**
         * @brief Predicts the output for a given input vector.
         * 
         * Performs a forward pass through the network to compute the output.
         * 
         * @param input The input vector.
         * @return The output vector.
         */
        std::vector<double> predict(const std::vector<double>& input);

        /**
         * @brief Trains the neural network using the provided training data.
         * 
         * Uses backpropagation and gradient descent to minimize the loss function.
         * 
         * @param inputs The training input data.
         * @param targets The expected output data corresponding to the inputs.
         * @param learning_rate The learning rate for gradient descent (default = 0.1).
         * @param epochs The number of training iterations (default = 10,000).
         */
        void train(
            const std::vector<std::vector<double>>& inputs, 
            const std::vector<std::vector<double>>& targets, 
            double learning_rate = 0.1, 
            int epochs = 10000
        );

        /**
         * @brief Computes the mean squared error (MSE) loss.
         * 
         * @param prediction The predicted output vector.
         * @param target The expected output vector.
         * @return The computed MSE loss.
         */
        double compute_mse_loss(
            const std::vector<double>& prediction, 
            const std::vector<double>& target
        );

        /**
         * @brief Computes the gradient of the loss with respect to the output layer.
         * 
         * @param prediction The predicted output vector.
         * @param target The expected output vector.
         * @return The gradient vector.
         */
        std::vector<double> compute_output_gradient(
            const std::vector<double>& prediction, 
            const std::vector<double>& target
        );

        /**
         * @brief Computes the accuracy of the network on a dataset.
         * 
         * @param inputs The input data.
         * @param targets The expected outputs.
         * @return The accuracy as a percentage (0.0 to 100.0).
         */
        double compute_accuracy(
            const std::vector<std::vector<double>>& inputs, 
            const std::vector<std::vector<double>>& targets
        );

        /**
         * @brief Determines whether a prediction is correct based on a target.
         * 
         * @param prediction The predicted output vector.
         * @param target The expected output vector.
         * @return True if the prediction is correct; otherwise, false.
         */
        bool is_correct_prediction(
            const std::vector<double>& prediction, 
            const std::vector<double>& target
        );

        /**
         * @brief Saves the current state of the neural network to a file.
         * 
         * @param filename The name of the file to save the model to.
         * 
         * @throws std::ios_base::failure if the file cannot be written.
         */
        void save_model(const std::string& filename);

        /**
         * @brief Loads a neural network from a saved model file.
         * 
         * @param filename The name of the file to load the model from.
         * @return A NeuralNetwork object initialized from the file data.
         * 
         * @throws std::ios_base::failure if the file cannot be read or is malformed.
         */
        static NeuralNetwork loadFromModel(const std::string& filename);
    };
}

#endif
