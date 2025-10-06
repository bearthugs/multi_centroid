#include <stdio.h>
#include <stdlib.h>
#include "quant_functions.h"

int main(int argc, char** argv) {

    // List of input and output files
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
    const char* output_files[] = {
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

    for (size_t i = 0; i < num_files; ++i) {
        uint32_t n, dim;
        float* data = read_fvecs(input_files[i], &n, &dim);
        if (!data) {
            fprintf(stderr, "Failed to read %s\n", input_files[i]);
            continue;
        }

        printf("Read %u vectors of dimension %u from %s\n", n, dim, input_files[i]);

        if (write_5bit(output_files[i], data, n, dim) != 0) {
            fprintf(stderr, "Error writing packed file %s.\n", output_files[i]);
            free(data);
            continue;
        }

        free(data);
        printf("5-bit quantisation and packing complete for %s.\n", input_files[i]);
    }

    printf("All 5-bit quantisation and packing complete.\n");

    return 0;
}
