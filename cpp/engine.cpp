#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>
#include <cstdint>
#include <immintrin.h>  // AVX

namespace py = pybind11;


// ------------------
// FP32 MatVec (BASELINE)
// ------------------
std::vector<float> matvec_flat(
    std::vector<float> W,
    std::vector<float> x,
    int rows,
    int cols
) {
    std::vector<float> result(rows, 0.0);

    for (int i = 0; i < rows; i++) {
        float sum = 0.0;
        for (int j = 0; j < cols; j++) {
            sum += W[i * cols + j] * x[j];
        }
        result[i] = sum;
    }

    return result;
}


// ------------------
// FP32 MatVec (LOOP UNROLLED)
// ------------------
std::vector<float> matvec_unrolled(
    std::vector<float> W,
    std::vector<float> x,
    int rows,
    int cols
) {
    std::vector<float> result(rows, 0.0);

    for (int i = 0; i < rows; i++) {
        float sum = 0.0;
        int j = 0;

        for (; j <= cols - 4; j += 4) {
            sum += W[i * cols + j] * x[j];
            sum += W[i * cols + j + 1] * x[j + 1];
            sum += W[i * cols + j + 2] * x[j + 2];
            sum += W[i * cols + j + 3] * x[j + 3];
        }

        for (; j < cols; j++) {
            sum += W[i * cols + j] * x[j];
        }

        result[i] = sum;
    }

    return result;
}


// ------------------
// FP32 MatVec (AVX SIMD)
// ------------------
std::vector<float> matvec_avx(
    std::vector<float> W,
    std::vector<float> x,
    int rows,
    int cols
) {
    std::vector<float> result(rows, 0.0);

    for (int i = 0; i < rows; i++) {
        __m256 sum_vec = _mm256_setzero_ps();
        int j = 0;

        for (; j <= cols - 8; j += 8) {
            __m256 w = _mm256_loadu_ps(&W[i * cols + j]);
            __m256 xv = _mm256_loadu_ps(&x[j]);

            __m256 prod = _mm256_mul_ps(w, xv);
            sum_vec = _mm256_add_ps(sum_vec, prod);
        }

        float temp[8];
        _mm256_storeu_ps(temp, sum_vec);

        float sum = temp[0] + temp[1] + temp[2] + temp[3] +
                    temp[4] + temp[5] + temp[6] + temp[7];

        for (; j < cols; j++) {
            sum += W[i * cols + j] * x[j];
        }

        result[i] = sum;
    }

    return result;
}


// ------------------
// Quantization
// ------------------
std::vector<int8_t> quantize_vector(std::vector<float> x, float scale) {
    std::vector<int8_t> q(x.size());

    for (size_t i = 0; i < x.size(); i++) {
        q[i] = static_cast<int8_t>(x[i] / scale);
    }

    return q;
}


// ------------------
// INT8 MatVec (FLAT)
// ------------------
std::vector<int> matvec_int8_flat(
    std::vector<int8_t> W,
    std::vector<int8_t> x,
    int rows,
    int cols
) {
    std::vector<int> result(rows, 0);

    for (int i = 0; i < rows; i++) {
        int sum = 0;

        for (int j = 0; j < cols; j++) {
            sum += W[i * cols + j] * x[j];
        }

        result[i] = sum;
    }

    return result;
}


// ------------------
// TERNARY (branch-free)
// ------------------
std::vector<float> matvec_ternary_flat(
    std::vector<int8_t> W,
    std::vector<float> x,
    int rows,
    int cols
) {
    std::vector<float> result(rows, 0.0);

    for (int i = 0; i < rows; i++) {
        float sum = 0.0;

        for (int j = 0; j < cols; j++) {
            sum += W[i * cols + j] * x[j];
        }

        result[i] = sum;
    }

    return result;
}


// ------------------
// Python Bindings
// ------------------
PYBIND11_MODULE(engine, m) {
    m.def("matvec_flat", &matvec_flat);
    m.def("matvec_unrolled", &matvec_unrolled);
    m.def("matvec_avx", &matvec_avx);

    m.def("quantize_vector", &quantize_vector);
    m.def("matvec_int8_flat", &matvec_int8_flat);

    m.def("matvec_ternary_flat", &matvec_ternary_flat);
}
