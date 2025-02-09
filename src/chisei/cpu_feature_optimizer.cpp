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

#include <chisei/cpu_feature_optimizer.hpp>

namespace chisei {

void CPUFeatureOptimizer::init_cpu_features(std::mt19937 gen) {
    #if defined(__RDRND__) && defined(__RDSEED__)
    uint32_t rand, seed, genSeed;

    while(_rdrand32_step(&rand) == 0);
    while(_rdseed32_step(&seed) == 0);
    while(_rdrand32_step(&genSeed) == 0);

    gen.seed(genSeed);
    #else
    gen.seed(static_cast<uint_fast32_t>(rand()));
    #endif
}

double CPUFeatureOptimizer::dot_product_fma(const double* a, const double* b, int size) {
    #ifdef __AVX__
    __m256d sum = _mm256_setzero_pd();
    for(int i = 0; i < size; i += 4) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        sum = _mm256_fmadd_pd(va, vb, sum);
    }

    double result[4];
    _mm256_storeu_pd(result, sum);

    return result[0] + result[1] + result[2] + result[3];
    #else
    double sum = 0.0;

    #pragma omp parallel for
    for(int i = 0; i < size; ++i)
        sum += a[i] * b[i];
    return sum;
    #endif
}

}
