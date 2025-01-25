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
 * @file ActivationFunctions.hpp
 * @author [Nathanne Isip](https://github.com/nthnn)
 * @brief Header file defining various activation functions and their derivatives 
 *        commonly used in neural networks.
 */
#ifndef CHISEI_ACTIVATION_FUNCTION_HPP
#define CHISEI_ACTIVATION_FUNCTION_HPP

#include <cmath>

namespace chisei {

    /**
     * @class ActivationFunctions
     * @brief Provides a collection of static methods for activation functions 
     *        and their derivatives.
     * 
     * This class includes commonly used activation functions such as Sigmoid, ReLU, 
     * and Tanh, along with their derivatives. All methods are static, making them 
     * accessible without instantiating the class.
     */
    class ActivationFunctions final {
    public:

        /**
         * @brief Computes the Sigmoid activation function.
         * 
         * The Sigmoid function is defined as:
         * \f[
         * f(x) = \frac{1}{1 + e^{-x}}
         * \f]
         * 
         * @param x The input value.
         * @return The computed Sigmoid value.
         */
        static constexpr inline double sigmoid_activation(double x) noexcept {
            return 1.0 / (1.0 + std::exp(-x));
        }

        /**
         * @brief Computes the derivative of the Sigmoid function.
         * 
         * The derivative of the Sigmoid function is:
         * \f[
         * f'(x) = x \cdot (1 - x)
         * \f]
         * 
         * This assumes that the input `x` is the output of the Sigmoid function.
         * 
         * @param x The input value (expected to be Sigmoid output).
         * @return The computed derivative value.
         */
        static constexpr inline double sigmoid_derivative(double x) noexcept {
            return x * (1.0 - x);
        }

        /**
         * @brief Computes the ReLU (Rectified Linear Unit) activation function.
         * 
         * The ReLU function is defined as:
         * \f[
         * f(x) = \max(0, x)
         * \f]
         * 
         * @param x The input value.
         * @return The computed ReLU value.
         */
        static constexpr inline double relu_activation(double x) noexcept {
            return x > 0.0 ? x : 0.0;
        }

        /**
         * @brief Computes the derivative of the ReLU function.
         * 
         * The derivative of the ReLU function is:
         * \f[
         * f'(x) = 
         * \begin{cases} 
         * 1 & \text{if } x > 0 \\
         * 0 & \text{if } x \leq 0
         * \end{cases}
         * \f]
         * 
         * @param x The input value.
         * @return The computed derivative value.
         */
        static constexpr inline double relu_derivative(double x) noexcept {
            return x > 0.0 ? 1.0 : 0.0;
        }

        /**
         * @brief Computes the Tanh (Hyperbolic Tangent) activation function.
         * 
         * The Tanh function is defined as:
         * \f[
         * f(x) = \tanh(x)
         * \f]
         * 
         * @param x The input value.
         * @return The computed Tanh value.
         */
        static constexpr inline double tanh_activation(double x) noexcept {
            return std::tanh(x);
        }

        /**
         * @brief Computes the derivative of the Tanh function.
         * 
         * The derivative of the Tanh function is:
         * \f[
         * f'(x) = 1 - x^2
         * \f]
         * 
         * This assumes that the input `x` is the output of the Tanh function.
         * 
         * @param x The input value (expected to be Tanh output).
         * @return The computed derivative value.
         */
        static constexpr inline double tanh_derivative(double x) noexcept {
            return 1.0 - x * x;
        }
    };
}

#endif
