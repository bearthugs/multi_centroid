#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../quant_functions.h"

// gcc -o tests/test_mse tests/test_quant_error.c quant_functions.c -lm

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
    return (float)(sqrt(mse / size));
    //return (float)(mse / size);
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

    // === Create CSV file ===
    FILE* csv = fopen("tests/quantisation_stats.csv", "w");
    if (!csv) {
        fprintf(stderr, "Error: could not create quantisation_stats.csv\n");
        return 1;
    }
    fprintf(csv, "dataset,n,dim,min_val,max_val,range,mse\n"); // CSV header

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

        // Print and compute range
        float range = header.max_val - header.min_val;
        printf("File %s: n=%u, dim=%u, min_val=%f, max_val=%f, range=%f\n",
                packed_files[i], header.n, header.dim,
                header.min_val, header.max_val, range);

        // Unpack
        float* unpacked = (float*)malloc(n * dim * sizeof(float));
        unpack_5bit(packed, n, dim, header.min_val, header.max_val, unpacked);

        // Compute error
        float mse = compute_mse(original, unpacked, n * dim);
        printf("sqrt(MSE) between original and unpacked: %f\n", mse);

        // === Write to CSV ===
        // Extract dataset name (just filename without path)
        const char* dataset_name = strrchr(packed_files[i], '/');
        dataset_name = dataset_name ? dataset_name + 1 : packed_files[i];

        fprintf(csv, "%s,%u,%u,%f,%f,%f,%f\n",
                dataset_name, header.n, header.dim,
                header.min_val, header.max_val, range, mse);

        // Cleanup
        free(original);
        free(packed);
        free(unpacked);
    }

    fclose(csv);
    printf("\nSaved results to tests/quantisation_stats.csv\n");
    return 0;
}
