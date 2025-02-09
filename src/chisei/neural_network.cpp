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

#include <chisei/neural_network.hpp>
#include <chisei/model_loader_exception.hpp>

namespace chisei {

NeuralNetwork::NeuralNetwork(
    const std::vector<size_t>& _layers,
    std::function<double(double)> _activation,
    std::function<double(double)> _activation_derivative
) : layer_sizes(_layers),
    weights(),
    biases(),
    activation(_activation),
    activation_derivative(_activation_derivative),
    rd(),
    gen(rd())
{
    CPUFeatureOptimizer::init_cpu_features(this->gen);

    for(size_t i = 1; i < layer_sizes.size(); ++i) {
        std::vector<std::vector<double>> layer_weights;

        for(size_t j = 0; j < layer_sizes[i-1]; ++j) {
            std::vector<double> neuron_weights(layer_sizes[i]);
            std::generate(
                neuron_weights.begin(),
                neuron_weights.end(), 
                [this]() {
                    return weight_dist(gen);
                }
            );

            layer_weights.emplace_back(neuron_weights);
        }
        weights.emplace_back(layer_weights);

        std::vector<double> layer_biases(layer_sizes[i]);
        std::generate(
            layer_biases.begin(),
            layer_biases.end(),
            [this]() {
                return weight_dist(gen);
            }
        );
        biases.emplace_back(layer_biases);
    }
}

NeuralNetwork::NeuralNetwork(const NeuralNetwork& other) :
    layer_sizes(std::move(other.layer_sizes)),
    weights(std::move(other.weights)),
    biases(std::move(other.biases)),
    activation(std::move(other.activation)),
    activation_derivative(std::move(other.activation_derivative)),
    rd(),
    gen(std::move(other.gen))
{ }

NeuralNetwork::~NeuralNetwork() {
    #pragma omp barrier
}

NeuralNetwork& NeuralNetwork::operator=(NeuralNetwork&& other) noexcept {
    if(this != &other) {
        this->layer_sizes = std::move(other.layer_sizes);
        this->weights = std::move(other.weights);
        this->biases = std::move(other.biases);
        this->activation = std::move(other.activation);
        this->activation_derivative = std::move(other.activation_derivative);
    }

    return *this;
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& input) {
    std::vector<double> layer_output = input;

    for(size_t layer = 0; layer < weights.size(); ++layer) {
        std::vector<double> next_layer_output(layer_sizes[layer + 1]);

        #pragma omp parallel for
        for(size_t j = 0; j < layer_sizes[layer + 1]; ++j) {
            double neuron_output = biases[layer][j];

            for(size_t i = 0; i < layer_sizes[layer]; ++i)
                neuron_output += layer_output[i] * weights[layer][i][j];
            next_layer_output[j] = this->activation(neuron_output);
        }

        layer_output = next_layer_output;
    }

    return layer_output;
}

void NeuralNetwork::train(
    const std::vector<std::vector<double>>& inputs, 
    const std::vector<std::vector<double>>& targets, 
    double learning_rate,
    int epochs
) {
    for(int epoch = 0; epoch < epochs; ++epoch) {
        for(size_t sample = 0; sample < inputs.size(); ++sample) {
            std::vector<std::vector<double>> layer_outputs;
            std::vector<double> current_input = inputs[sample];

            layer_outputs.emplace_back(current_input);
            for(size_t layer = 0; layer < weights.size(); ++layer) {
                std::vector<double> next_layer_output(layer_sizes[layer + 1]);

                for(size_t j = 0; j < layer_sizes[layer + 1]; ++j) {
                    double neuron_output = biases[layer][j];

                    for(size_t i = 0; i < layer_sizes[layer]; ++i)
                        neuron_output += current_input[i] * weights[layer][i][j];
                    next_layer_output[j] = this->activation(neuron_output);
                }

                layer_outputs.emplace_back(next_layer_output);
                current_input = next_layer_output;
            }

            std::vector<std::vector<double>> gradients(weights.size());
            std::vector<double> output_gradient(layer_sizes.back());

            for(size_t j = 0; j < layer_sizes.back(); ++j) {
                double output = layer_outputs.back()[j];
                output_gradient[j] = (output - targets[sample][j]) *
                    this->activation_derivative(output);
            }
            gradients.back() = output_gradient;

            for(int layer = (int) weights.size() - 2; layer >= 0; --layer) {
                std::vector<double> layer_gradient(
                    layer_sizes[static_cast<size_t>(layer + 1)]
                );

                for(size_t j = 0; j < layer_sizes[static_cast<size_t>(layer + 1)]; ++j) {
                    double gradient_sum = 0.0;

                    for(size_t k = 0; k < layer_sizes[static_cast<size_t>(layer + 2)]; ++k)
                        gradient_sum += gradients[static_cast<size_t>(layer + 1)][k] *
                            weights[static_cast<size_t>(layer + 1)][j][k];

                    double layer_output = layer_outputs[static_cast<size_t>(layer + 1)][j];
                    layer_gradient[j] = gradient_sum *
                        this->activation_derivative(layer_output);
                }
                
                gradients[static_cast<size_t>(layer)] = layer_gradient;
            }

            for(size_t layer = 0; layer < weights.size(); ++layer) {
                for(size_t i = 0; i < layer_sizes[layer]; ++i)
                    for(size_t j = 0; j < layer_sizes[layer + 1]; ++j)
                        weights[layer][i][j] -= learning_rate * 
                            gradients[layer][j] * layer_outputs[layer][i];

                for(size_t j = 0; j < layer_sizes[layer + 1]; ++j)
                    biases[layer][j] -= learning_rate * gradients[layer][j];
            }
        }
    }
}

double compute_mse_loss(const std::vector<double>& prediction, 
    const std::vector<double>& target) {
    double total_loss = 0.0;

    #pragma omp parallel for
    for(size_t i = 0; i < prediction.size(); ++i) {
        double diff = prediction[i] - target[i];
        total_loss += diff * diff;
    }

    return total_loss / (double) prediction.size();
}

std::vector<double> compute_output_gradient(
    const std::vector<double>& prediction, 
    const std::vector<double>& target
) {
    std::vector<double> gradient(prediction.size());

    for(size_t i = 0; i < prediction.size(); ++i)
        gradient[i] = 2 * (prediction[i] - target[i]);

    return gradient;
}

double NeuralNetwork::compute_accuracy(const std::vector<std::vector<double>>& inputs, 
    const std::vector<std::vector<double>>& targets) {
    int correct_predictions = 0;

    for(size_t i = 0; i < inputs.size(); ++i) {
        std::vector<double> prediction = predict(inputs[i]);

        if(is_correct_prediction(prediction, targets[i]))
            ++correct_predictions;
    }

    return static_cast<double>(correct_predictions) / (double) inputs.size();
}

bool NeuralNetwork::is_correct_prediction(
    const std::vector<double>& prediction, 
    const std::vector<double>& target
) {
    size_t pred_max_idx = static_cast<size_t>(
        std::max_element(
            prediction.begin(),
            prediction.end()
        ) - prediction.begin()
    );
    size_t target_max_idx = static_cast<size_t>(
        std::max_element(
            target.begin(),
            target.end()
        ) - target.begin()
    );

    return pred_max_idx == target_max_idx;
}

void NeuralNetwork::save_model(const std::string& filename) {
    std::string final_filename = filename;
    if(final_filename.substr(final_filename.size() - 7) != ".chisei")
        final_filename += ".chisei";

    std::ofstream file(final_filename, std::ios::binary);
    if(!file)
        throw ModelLoaderException("Failed to open *.chisei file for saving the model.");

    const char magic[] = "CS";
    file.write(magic, sizeof(magic) - 1);

    size_t layer_count = layer_sizes.size();
    file.write(reinterpret_cast<char*>(&layer_count), sizeof(layer_count));

    for(size_t i = 0; i < layer_count; ++i) {
        size_t size = layer_sizes[i];
        file.write(reinterpret_cast<char*>(&size), sizeof(size));
    }

    for(size_t layer = 0; layer < weights.size(); ++layer)
        for(size_t i = 0; i < layer_sizes[layer]; ++i)
            file.write(
                reinterpret_cast<char*>(weights[layer][i].data()),
                static_cast<std::streamsize>(weights[layer][i].size() * sizeof(double))
            );

    for(size_t layer = 0; layer < biases.size(); ++layer)
        file.write(
            reinterpret_cast<char*>(biases[layer].data()),
            static_cast<std::streamsize>(biases[layer].size() * sizeof(double))
        );

    file.close();
}

NeuralNetwork NeuralNetwork::loadFromModel(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if(!file.is_open())
        throw ModelLoaderException("Failed to open file for loading model.");

    char magic[2] = {0};
    file.read(magic, sizeof(magic));
    if(magic[0] != 'C' || magic[1] != 'S')
        throw ModelLoaderException("Invalid *.chisei file format, missing magic bytes.");

    size_t num_layers;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));

    std::vector<size_t> layer_sizes(num_layers);
    file.read(
        reinterpret_cast<char*>(layer_sizes.data()),
        static_cast<std::streamsize>(num_layers * sizeof(size_t))
    );

    NeuralNetwork network(
        layer_sizes,
        ActivationFunctions::sigmoid_activation,
        ActivationFunctions::sigmoid_derivative
    );

    network.weights.resize(num_layers - 1);
    network.biases.resize(num_layers - 1);

    for(size_t layer = 0; layer < num_layers - 1; ++layer) {
        network.weights[layer].resize(layer_sizes[layer]);

        for(size_t i = 0; i < layer_sizes[layer]; ++i) {
            network.weights[layer][i].resize(layer_sizes[layer + 1]);
            file.read(
                reinterpret_cast<char*>(network.weights[layer][i].data()),
                static_cast<std::streamsize>(layer_sizes[layer + 1] * sizeof(double))
            );
        }
    }

    for(size_t layer = 0; layer < num_layers - 1; ++layer) {
        network.biases[layer].resize(layer_sizes[layer + 1]);
        file.read(
            reinterpret_cast<char*>(network.biases[layer].data()),
            static_cast<std::streamsize>(layer_sizes[layer + 1] * sizeof(double))
        );
    }

    file.close();
    return network;
}

}
