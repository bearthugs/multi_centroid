#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "quant_functions.h"

float* read_fvecs(const char* filename, uint32_t* n, uint32_t* dim);

void unpack_5bit(const uint8_t* packed, uint32_t n, uint32_t dim, float min_val, float max_val, float* output) {
    uint32_t bit_pos = 0;
    for (uint32_t i = 0; i < n * dim; i++) {
        uint8_t qval = 0;
        for (uint8_t bit = 0; bit < BIT_DEPTH; bit++) {
            if (packed[bit_pos / 8] & (1 << (bit_pos % 8))) {
                qval |= (1 << bit);
            }
            bit_pos++;
        }
        // Rescale back to float
        float normalized = (float)qval / (LEVELS - 1);
        output[i] = normalized * (max_val - min_val) + min_val;
    }
}

float compute_mse(const float* a, const float* b, uint32_t size) {
    double mse = 0.0;
    for (uint32_t i = 0; i < size; i++) {
        double diff = a[i] - b[i];
        mse += diff * diff;
    }
    return (float)(mse / size);
}

int main() {
    const char* input_files[] = {
         "reduced_data/coco-i2i-512-angular_train_reduced.fvecs",
        "reduced_data/fashion-mnist-784-euclidean_train_reduced.fvecs",
        "reduced_data/gist-960-euclidean_train_reduced.fvecs",
        "reduced_data/glove-25-angular_train_reduced.fvecs",
        "reduced_data/glove-50-angular_train_reduced.fvecs",
        "reduced_data/glove-100-angular_train_reduced.fvecs",
        "reduced_data/glove-200-angular_train_reduced.fvecs",
        "reduced_data/lastfm-64-dot_train_reduced.fvecs",
        "reduced_data/mnist-784-euclidean_train_reduced.fvecs",
        "reduced_data/nytimes-256-angular_train_reduced.fvecs",
        "reduced_data/sift-128-euclidean_train_reduced.fvecs"
    };

    const char* packed_files[] = {
       "quantised_data/coco-i2i-512-angular_train_reduced.5bit",
        "quantised_data/fashion-mnist-784-euclidean_train_reduced.5bit",
        "quantised_data/gist-960-euclidean_train_reduced.5bit",
        "quantised_data/glove-25-angular_train_reduced.5bit",
        "quantised_data/glove-50-angular_train_reduced.5bit",
        "quantised_data/glove-100-angular_train_reduced.5bit",
        "quantised_data/glove-200-angular_train_reduced.5bit",
        "quantised_data/lastfm-64-dot_train_reduced.5bit",
        "quantised_data/mnist-784-euclidean_train_reduced.5bit",
        "quantised_data/nytimes-256-angular_train_reduced.5bit",
        "quantised_data/sift-128-euclidean_train_reduced.5bit"
    };

    const size_t num_files = sizeof(input_files) / sizeof(input_files[0]);

    for (size_t i = 0; i < num_files; i++) {
        uint32_t n, dim;

        // Read original vectors
        float* original = read_fvecs(input_files[i], &n, &dim);
        if (!original) {
            fprintf(stderr, "Failed to read original file: %s\n", input_files[i]);
            continue;
        }

        // Read packed data
        header_t header;
        uint8_t* packed = read_packed(packed_files[i], &header);
        if (!packed) {
            fprintf(stderr, "Failed to read packed file: %s\n", packed_files[i]);
            free(original);
            continue;
        }

        if (header.n != n || header.dim != dim) {
            fprintf(stderr, "Mismatch in dimensions for %s\n", input_files[i]);
            free(original);
            free(packed);
            continue;
        }

        // Unpack
        float* unpacked = (float*)malloc(n * dim * sizeof(float));
        unpack_5bit(packed, n, dim, header.min_val, header.max_val, unpacked);

        // Compute error
        float mse = compute_mse(original, unpacked, n * dim);
        printf("File: %s\n", input_files[i]);
        printf("MSE between original and unpacked: %f\n", mse);

        free(original);
        free(packed);
        free(unpacked);
    }

    return 0;
}
