#include <stdio.h>
#include <stdlib.h>
#include "quant_functions.h"

/*
COMPILING
    gcc -O2 5bit_quantisation/quant_main.c 5bit_quantisation/quant_functions.c -lm -o quantise

RUNNING
    ./quantise
*/

int main(int argc, char** argv) {

    // List of input and output files
    const char* input_files[] = {
        "reduced_data/train/coco-i2i-512-angular_train_reduced.fvecs",
        "reduced_data/train/fashion-mnist-784-euclidean_train_reduced.fvecs",
        "reduced_data/train/gist-960-euclidean_train_reduced.fvecs",
        "reduced_data/train/glove-25-angular_train_reduced.fvecs",
        "reduced_data/train/glove-50-angular_train_reduced.fvecs",
        "reduced_data/train/glove-100-angular_train_reduced.fvecs",
        "reduced_data/train/glove-200-angular_train_reduced.fvecs",
        "reduced_data/train/lastfm-64-dot_train_reduced.fvecs",
        "reduced_data/train/mnist-784-euclidean_train_reduced.fvecs",
        "reduced_data/train/nytimes-256-angular_train_reduced.fvecs",
        "reduced_data/train/sift-128-euclidean_train_reduced.fvecs",
        "reduced_data/test/coco-i2i-512-angular_test_reduced.fvecs",
        "reduced_data/test/fashion-mnist-784-euclidean_test_reduced.fvecs",
        "reduced_data/test/gist-960-euclidean_test_reduced.fvecs",
        "reduced_data/test/glove-25-angular_test_reduced.fvecs",
        "reduced_data/test/glove-50-angular_test_reduced.fvecs",
        "reduced_data/test/glove-100-angular_test_reduced.fvecs",
        "reduced_data/test/glove-200-angular_test_reduced.fvecs",
        "reduced_data/test/lastfm-64-dot_test_reduced.fvecs",
        "reduced_data/test/mnist-784-euclidean_test_reduced.fvecs",
        "reduced_data/test/nytimes-256-angular_test_reduced.fvecs",
        "reduced_data/test/sift-128-euclidean_test_reduced.fvecs"
    };
    const char* output_files[] = {
        "quantised_data/train/coco-i2i-512-angular_train_reduced.5bit",
        "quantised_data/train/fashion-mnist-784-euclidean_train_reduced.5bit",
        "quantised_data/train/gist-960-euclidean_train_reduced.5bit",
        "quantised_data/train/glove-25-angular_train_reduced.5bit",
        "quantised_data/train/glove-50-angular_train_reduced.5bit",
        "quantised_data/train/glove-100-angular_train_reduced.5bit",
        "quantised_data/train/glove-200-angular_train_reduced.5bit",
        "quantised_data/train/lastfm-64-dot_train_reduced.5bit",
        "quantised_data/train/mnist-784-euclidean_train_reduced.5bit",
        "quantised_data/train/nytimes-256-angular_train_reduced.5bit",
        "quantised_data/train/sift-128-euclidean_train_reduced.5bit",
        "quantised_data/test/coco-i2i-512-angular_test_reduced.5bit",
        "quantised_data/test/fashion-mnist-784-euclidean_test_reduced.5bit",
        "quantised_data/test/gist-960-euclidean_test_reduced.5bit",
        "quantised_data/test/glove-25-angular_test_reduced.5bit",
        "quantised_data/test/glove-50-angular_test_reduced.5bit",
        "quantised_data/test/glove-100-angular_test_reduced.5bit",
        "quantised_data/test/glove-200-angular_test_reduced.5bit",
        "quantised_data/test/lastfm-64-dot_test_reduced.5bit",
        "quantised_data/test/mnist-784-euclidean_test_reduced.5bit",
        "quantised_data/test/nytimes-256-angular_test_reduced.5bit",
        "quantised_data/test/sift-128-euclidean_test_reduced.5bit"
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
