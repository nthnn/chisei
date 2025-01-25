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
 * @file CPUFeatureOptimizer.hpp
 * @author [Nathanne Isip](https://github.com/nthnn)
 * @brief Header file for optimizing CPU-specific features such as hardware acceleration
 *        and advanced vector extensions.
 */
#ifndef CHISEI_CPU_FEATURE_OPTIMIZER_HPP
#define CHISEI_CPU_FEATURE_OPTIMIZER_HPP

#include <random>

#if(defined(__RDRND__) && defined(__RDSEED__)) || defined(__AVX__)
#   include <immintrin.h>
#endif

namespace chisei {

    /**
     * @class CPUFeatureOptimizer
     * @brief Provides utilities to optimize computations using CPU-specific features, such as FMA and AVX.
     * 
     * This class includes methods to initialize CPU-specific features and perform optimized 
     * mathematical operations such as dot products. It utilizes hardware acceleration 
     * where available for improved performance.
     */
    class CPUFeatureOptimizer {
    public:

        /**
         * @brief Initializes CPU-specific features for optimization.
         * 
         * This method is used to initialize and configure CPU features based on 
         * the available hardware capabilities. It can utilize features such as 
         * RDRAND, RDSEED, and AVX if supported by the CPU.
         * 
         * @param gen A random number generator of type `std::mt19937`, which may 
         *            be used internally for testing or feature initialization.
         */
        static void init_cpu_features(std::mt19937 gen);

        /**
         * @brief Computes the dot product of two arrays using FMA (Fused Multiply-Add) instructions.
         * 
         * If FMA or AVX instructions are available, this method will leverage hardware 
         * acceleration for computing the dot product of two arrays. FMA reduces rounding 
         * errors and improves performance by combining multiplication and addition in a 
         * single instruction.
         * 
         * @param a Pointer to the first array of double-precision floating-point numbers.
         * @param b Pointer to the second array of double-precision floating-point numbers.
         * @param size The size of the arrays (number of elements in each array).
         * 
         * @return The computed dot product of the two arrays.
         * 
         * @note The arrays `a` and `b` must have at least `size` elements to avoid undefined behavior.
         */
        static inline double dot_product_fma(
            const double* a,
            const double* b,
            int size
        );
    };
}

#endif
