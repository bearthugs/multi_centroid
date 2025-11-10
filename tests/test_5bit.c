#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../quant_functions.h"

/*
COMPILING:
    gcc -O2 tests/test_5bit.c quant_functions.c -lm -o test_5bit

RUNNING FOR EACH DATASET:
    ./test_clustering quantised_data/coco-i2i-512-angular_train_reduced.5bit
    ./test_clustering quantised_data/fashion-mnist-784-euclidean_train_reduced.5bit
    ./test_clustering quantised_data/gist-960-euclidean_train_reduced.5bit
    ./test_clustering quantised_data/glove-25-angular_train_reduced.5bit
    ./test_clustering quantised_data/glove-50-angular_train_reduced.5bit
    ./test_clustering quantised_data/glove-100-angular_train_reduced.5bit
    ./test_clustering quantised_data/glove-200-angular_train_reduced.5bit
    ./test_clustering quantised_data/lastfm-64-dot_train_reduced.5bit
    ./test_clustering quantised_data/mnist-784-euclidean_train_reduced.5bit
    ./test_clustering quantised_data/nytimes-256-angular_train_reduced.5bit
    ./test_clustering quantised_data/sift-128-euclidean_train_reduced.5bit
*/


// Forward declaration from gmm_cluster_runner.c
static float** dequantize_all(const uint8_t* packed, const header_t* header);

// Simple Euclidean distance for float vectors
float l2_distance(const float* a, const float* b, uint32_t dim) {
    float acc = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        acc += diff * diff;
    }
    return sqrtf(acc);
}

// Dequantize helper copied from gmm_cluster_runner.c
static float** dequantize_all(const uint8_t* packed, const header_t* header) {
    uint32_t n = header->n, dim = header->dim;
    float min_val = header->min_val;
    float max_val = header->max_val;
    float range = max_val - min_val;

    float** data = (float**)malloc(n * sizeof(float*));
    for (uint32_t i = 0; i < n; i++)
        data[i] = (float*)calloc(dim, sizeof(float));

    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t d = 0; d < dim; d++) {
            uint32_t bit_pos = (i * dim + d) * BIT_DEPTH;
            uint8_t qval = 0;
            for (uint8_t b = 0; b < BIT_DEPTH; b++) {
                if (packed[(bit_pos + b) / 8] & (1 << ((bit_pos + b) % 8))) {
                    qval |= (1 << b);
                }
            }
            data[i][d] = min_val + (qval / (float)(LEVELS - 1)) * range;
        }
    }
    return data;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <quantised_dataset.5bit>\n", argv[0]);
        return 1;
    }

    const char* filename = argv[1];
    printf("=== Testing %s ===\n", filename);

    // 1️⃣ Read header
    header_t header;
    uint8_t* packed = read_packed(filename, &header);
    if (!packed) {
        fprintf(stderr, "Failed to read .5bit file\n");
        return 1;
    }

    printf("Header:\n");
    printf("  n        = %u\n", header.n);
    printf("  dim      = %u\n", header.dim);
    printf("  min_val  = %.6f\n", header.min_val);
    printf("  max_val  = %.6f\n", header.max_val);

    // 2️⃣ File size sanity check
    FILE* f = fopen(filename, "rb");
    fseek(f, 0, SEEK_END);
    long actual_size = ftell(f);
    fclose(f);

    long expected_size = 16 + ((long)header.n * header.dim * BIT_DEPTH + 7) / 8;
    printf("File size check:\n");
    printf("  Actual   = %ld bytes\n", actual_size);
    printf("  Expected ≈ %ld bytes\n", expected_size);
    if (llabs(actual_size - expected_size) < 16)
        printf("  ✅ Size check PASSED\n");
    else
        printf("  ⚠️  Size mismatch (file may be incomplete)\n");

    // 3️⃣ Dequantize first few vectors
    printf("\nDequantizing first 2 vectors...\n");
    float** data = dequantize_all(packed, &header);
    for (uint32_t i = 0; i < 2 && i < header.n; i++) {
        printf("Vector[%u]: ", i);
        for (uint32_t d = 0; d < (header.dim < 8 ? header.dim : 8); d++) {
            printf("%.4f ", data[i][d]);
        }
        if (header.dim > 8) printf("...");
        printf("\n");
    }

    // 4️⃣ Quantised distance check between first 2 vectors
    if (header.n >= 2) {
        float dq_dist = l2_distance(data[0], data[1], header.dim);
        float q_dist = distance(packed, 0, 1, &header, DIST_EUCLIDEAN);
        printf("\nDistance Check (vector 0 vs 1):\n");
        printf("  Dequantized L2 = %.6f\n", dq_dist);
        printf("  Quantised L2   = %.6f\n", q_dist);
        printf("  Δ = %.6f\n", fabsf(dq_dist - q_dist));
    }

    // 5️⃣ Basic quantisation fidelity check
    float minv = header.min_val, maxv = header.max_val, range = maxv - minv;
    float max_err = 0.0f, mean_err = 0.0f;
    uint64_t count = 0;

    for (uint32_t i = 0; i < (header.n < 10 ? header.n : 10); i++) {
        for (uint32_t d = 0; d < header.dim; d++) {
            float val = data[i][d];
            uint8_t qval = (uint8_t)(((val - minv) / range) * (LEVELS - 1) + 0.5f);
            float recon = minv + (qval / (float)(LEVELS - 1)) * range;
            float err = fabsf(val - recon);
            mean_err += err;
            if (err > max_err) max_err = err;
            count++;
        }
    }

    mean_err /= count;
    printf("\nQuantisation Fidelity Check (first 10 vectors):\n");
    printf("  Mean abs error = %.6e\n", mean_err);
    printf("  Max abs error  = %.6e\n", max_err);

    printf("\n✅ All sanity checks completed.\n");

    // Free memory
    for (uint32_t i = 0; i < header.n; i++) free(data[i]);
    free(data);
    free(packed);
    return 0;
}
