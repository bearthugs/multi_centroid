#include "../quant_functions.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

// gcc quant_functions.c tests/test_quant_functions.c -o tests/test_quant_functions -lm

// Manual quantisation helper: same logic as write_5bit()
static inline uint8_t quantise(float v, float min_val, float max_val) {
    float norm = (v - min_val) / (max_val - min_val);
    uint8_t q = (uint8_t)(norm * (LEVELS - 1) + 0.5f);
    if (q >= LEVELS) q = LEVELS - 1;
    return q;
}

// Manual dequantisation helper: same logic as distance()
static inline float dequantise(uint8_t q, float min_val, float max_val) {
    float range = max_val - min_val;
    return min_val + (range * q) / (LEVELS - 1);
}

// Manual Euclidean computation on quantised data
float manual_quant_euclidean(const float* a, const float* b, uint32_t dim,
                             float min_val, float max_val) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        uint8_t q1 = quantise(a[i], min_val, max_val);
        uint8_t q2 = quantise(b[i], min_val, max_val);
        float v1 = dequantise(q1, min_val, max_val);
        float v2 = dequantise(q2, min_val, max_val);
        float diff = v1 - v2;
        sum += diff * diff;
    }
    return sqrtf(sum);
}

// Manual cosine computation on quantised data
float manual_quant_cosine(const float* a, const float* b, uint32_t dim,
                          float min_val, float max_val) {
    float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        uint8_t q1 = quantise(a[i], min_val, max_val);
        uint8_t q2 = quantise(b[i], min_val, max_val);
        float v1 = dequantise(q1, min_val, max_val);
        float v2 = dequantise(q2, min_val, max_val);
        dot += v1 * v2;
        norm1 += v1 * v1;
        norm2 += v2 * v2;
    }
    float denom = sqrtf(norm1) * sqrtf(norm2);
    if (denom == 0.0f) return 0.0f;
    return 1.0f - (dot / denom);
}

int main() {
    const char* out_filename = "test_vectors.5bit";
    uint32_t n = 3, dim = 4;
    float data[] = {
        0.1f, 0.2f, 0.3f, 0.4f,     // vec 0
        0.4f, 0.5f, 0.6f, 0.7f,     // vec 1
        -0.2f, 0.0f, 0.2f, 0.4f     // vec 2
    };

    printf("=== Writing test vectors to 5-bit file ===\n");
    if (write_5bit(out_filename, data, n, dim) != 0) {
        fprintf(stderr, "Error: write_5bit failed.\n");
        return 1;
    }

    header_t header;
    uint8_t* packed = read_packed(out_filename, &header);
    if (!packed) {
        fprintf(stderr, "Error: read_packed failed.\n");
        return 1;
    }

    printf("Header: n=%u, dim=%u, min=%.3f, max=%.3f\n",
           header.n, header.dim, header.min_val, header.max_val);

    // --- Verification ---
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = i + 1; j < n; j++) {
            float d_func_euc = distance(packed, i, j, &header, DIST_EUCLIDEAN);
            float d_func_cos = distance(packed, i, j, &header, DIST_COSINE);

            float d_manual_euc = manual_quant_euclidean(&data[i * dim], &data[j * dim], dim,
                                                        header.min_val, header.max_val);
            float d_manual_cos = manual_quant_cosine(&data[i * dim], &data[j * dim], dim,
                                                     header.min_val, header.max_val);

            printf("\nVector pair (%u, %u):\n", i, j);
            printf("  Distance() Euclidean: %.8f\n", d_func_euc);
            printf("  Manual     Euclidean: %.8f\n", d_manual_euc);
            printf("  Distance() Cosine:    %.8f\n", d_func_cos);
            printf("  Manual     Cosine:    %.8f\n", d_manual_cos);

            // Use strict float equality (within float rounding tolerance)
            assert(fabsf(d_func_euc - d_manual_euc) < 1e-6f);
            assert(fabsf(d_func_cos - d_manual_cos) < 1e-6f);
        }
    }

    printf("\n✅ Exact quantised-distance verification passed!\n");

    free(packed);
    return 0;
}
