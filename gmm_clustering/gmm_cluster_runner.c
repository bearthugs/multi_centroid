#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "../quant_functions.h"
#include "gmm_cluster_runner.h"

#define K 8                 // Number of GMM clusters
#define MAX_ITER 50         // EM iterations
#define EPSILON 1e-6

/*
COMPILING:
    gcc -O2 gmm_clustering/gmm_cluster_runner.c quant_functions.c -lm -o gmm_cluster_runner

RUNNING FOR EACH DATASET
    ./gmm_cluster_runner quantised_data/coco-i2i-512-angular_train_reduced.5bit
    ./gmm_cluster_runner quantised_data/fashion-mnist-784-euclidean_train_reduced.5bit
    ./gmm_cluster_runner quantised_data/gist-960-euclidean_train_reduced.5bit
    ./gmm_cluster_runner quantised_data/glove-25-angular_train_reduced.5bit
    ./gmm_cluster_runner quantised_data/glove-50-angular_train_reduced.5bit
    ./gmm_cluster_runner quantised_data/glove-100-angular_train_reduced.5bit
    ./gmm_cluster_runner quantised_data/glove-200-angular_train_reduced.5bit
    ./gmm_cluster_runner quantised_data/lastfm-64-dot_train_reduced.5bit
    ./gmm_cluster_runner quantised_data/mnist-784-euclidean_train_reduced.5bit
    ./gmm_cluster_runner quantised_data/nytimes-256-angular_train_reduced.5bit
    ./gmm_cluster_runner quantised_data/sift-128-euclidean_train_reduced.5bit
*/

// Allocate 2D array
static float** alloc_2d(uint32_t rows, uint32_t cols) {
    float** arr = (float**)malloc(rows * sizeof(float*));
    for (uint32_t i = 0; i < rows; i++) {
        arr[i] = (float*)calloc(cols, sizeof(float));
    }
    return arr;
}

// Free 2D array
static void free_2d(float** arr, uint32_t rows) {
    for (uint32_t i = 0; i < rows; i++) free(arr[i]);
    free(arr);
}

// Dequantize a packed dataset to float[n][dim]
static float** dequantize_all(const uint8_t* packed, const header_t* header) {
    uint32_t n = header->n, dim = header->dim;
    float min_val = header->min_val;
    float max_val = header->max_val;
    float range = max_val - min_val;

    float** data = alloc_2d(n, dim);

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

// Compute squared L2 distance between two vectors
static float l2_sq(const float* a, const float* b, uint32_t dim) {
    float acc = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        acc += diff * diff;
    }
    return acc;
}

// GMM initialization (KMeans++-like seeding)
static void initialize_gmm(float** data, uint32_t n, uint32_t dim,
                           float** means, float* weights) {
    // Choose first centroid randomly
    srand(42);
    uint32_t first = rand() % n;
    memcpy(means[0], data[first], dim * sizeof(float));

    // For others, pick farthest samples
    for (uint32_t k = 1; k < K; k++) {
        float max_dist = -1.0f;
        uint32_t best = 0;
        for (uint32_t i = 0; i < n; i++) {
            float min_d = INFINITY;
            for (uint32_t j = 0; j < k; j++) {
                float d = l2_sq(data[i], means[j], dim);
                if (d < min_d) min_d = d;
            }
            if (min_d > max_dist) {
                max_dist = min_d;
                best = i;
            }
        }
        memcpy(means[k], data[best], dim * sizeof(float));
    }

    for (uint32_t k = 0; k < K; k++) weights[k] = 1.0f / K;
}

// EM algorithm
static void run_gmm(float** data, uint32_t n, uint32_t dim,
                    float** means, float* weights, float** resp) {
    float var = 1.0f; // shared variance for simplicity

    for (uint32_t iter = 0; iter < MAX_ITER; iter++) {
        // E-step
        for (uint32_t i = 0; i < n; i++) {
            float sum_r = 0.0f;
            for (uint32_t k = 0; k < K; k++) {
                float dist = l2_sq(data[i], means[k], dim);
                float exponent = expf(-0.5f * dist / var);
                resp[i][k] = weights[k] * exponent;
                sum_r += resp[i][k];
            }
            for (uint32_t k = 0; k < K; k++) resp[i][k] /= (sum_r + EPSILON);
        }

        // M-step
        float Nk[K] = {0};
        for (uint32_t k = 0; k < K; k++) {
            for (uint32_t i = 0; i < n; i++) Nk[k] += resp[i][k];
        }

        // Update means
        for (uint32_t k = 0; k < K; k++) {
            memset(means[k], 0, dim * sizeof(float));
            for (uint32_t i = 0; i < n; i++) {
                for (uint32_t d = 0; d < dim; d++) {
                    means[k][d] += resp[i][k] * data[i][d];
                }
            }
            for (uint32_t d = 0; d < dim; d++) {
                means[k][d] /= (Nk[k] + EPSILON);
            }
        }

        // Update weights
        for (uint32_t k = 0; k < K; k++) {
            weights[k] = Nk[k] / n;
        }

        if (iter % 10 == 0)
            printf("Iteration %u complete\n", iter);
    }
}

// Assign each vector to its most probable cluster
static uint32_t* assign_clusters(float** resp, uint32_t n) {
    uint32_t* cluster_ids = (uint32_t*)malloc(n * sizeof(uint32_t));
    for (uint32_t i = 0; i < n; i++) {
        uint32_t best = 0;
        float max_r = resp[i][0];
        for (uint32_t k = 1; k < K; k++) {
            if (resp[i][k] > max_r) {
                max_r = resp[i][k];
                best = k;
            }
        }
        cluster_ids[i] = best;
    }
    return cluster_ids;
}

// Save GMM index
static void save_index(const char* filename, uint32_t* cluster_ids,
                       uint32_t n, uint32_t dim) {
    FILE* f = fopen(filename, "wb");
    if (!f) { perror("fopen"); return; }

    fwrite(&n, sizeof(uint32_t), 1, f);
    fwrite(&dim, sizeof(uint32_t), 1, f);
    uint32_t k_val = K;
    fwrite(&k_val, sizeof(uint32_t), 1, f);
    fwrite(cluster_ids, sizeof(uint32_t), n, f);

    fclose(f);
    printf("Saved GMM index to %s\n", filename);
}

void compute_cluster_probabilities(const float* query, const gmm_params_t* gmm, float* probs) {
    float var = gmm->variance;
    float sum_p = 0.0f;
    for (uint32_t k = 0; k < K; k++) {
        float dist = 0.0f;
        for (uint32_t d = 0; d < gmm->dim; d++) {
            float diff = query[d] - gmm->means[k][d];
            dist += diff * diff;
        }
        float p = gmm->weights[k] * expf(-0.5f * dist / var);
        probs[k] = p;
        sum_p += p;
    }
    for (uint32_t k = 0; k < K; k++) probs[k] /= (sum_p + EPSILON);
}


int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <quantised_dataset.5bit>\n", argv[0]);
        return 1;
    }
    clock_t start = clock();
    const char* in_filename = argv[1];
    header_t header;
    uint8_t* packed = read_packed(in_filename, &header);
    if (!packed) return 1;

    printf("Loaded %u vectors of dim %u from %s\n",
           header.n, header.dim, in_filename);

    // Dequantize
    float** data = dequantize_all(packed, &header);
    free(packed);

    // Allocate GMM arrays
    float** means = alloc_2d(K, header.dim);
    float* weights = (float*)calloc(K, sizeof(float));
    float** resp = alloc_2d(header.n, K);

    // Initialize and run GMM
    initialize_gmm(data, header.n, header.dim, means, weights);
    run_gmm(data, header.n, header.dim, means, weights, resp);

    // Assign clusters and save index
    uint32_t* cluster_ids = assign_clusters(resp, header.n);
    printf("Cluster assignment took %.2f s\n", (clock() - start)/ (float) CLOCKS_PER_SEC);

    char out_filename[512];
    snprintf(out_filename, sizeof(out_filename),
             "gmm_indexes/%s.index", strrchr(in_filename, '/') ?
             strrchr(in_filename, '/') + 1 : in_filename);
    
    save_index(out_filename, cluster_ids, header.n, header.dim);

    // Free
    free_2d(data, header.n);
    free_2d(means, K);
    free(weights);
    free_2d(resp, header.n);
    free(cluster_ids);

    return 0;
}
