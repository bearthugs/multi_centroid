#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include "../quant_functions.h"
#include "../gmm_clustering/utils.h"

/*
COMPILING:
    gcc -O2 tests/test_clustering.c quant_functions.c gmm_clustering/utils.c -lm -o test_clustering



RUNNING FOR EACH DATASET:

    ./test_clustering \
    quantised_data/coco-i2i-512-angular_train_reduced.5bit \
    gmm_indexes/coco-i2i-512-angular_train_reduced.5bit.index \
    gmm_indexes/coco-i2i-512-angular_train_reduced.5bit.gmm

    ./test_clustering \
    quantised_data/fashion-mnist-784-euclidean_train_reduced.5bit \
    gmm_indexes/fashion-mnist-784-euclidean_train_reduced.5bit.index \
    gmm_indexes/fashion-mnist-784-euclidean_train_reduced.5bit.gmm

    ./test_clustering \
    quantised_data/gist-960-euclidean_train_reduced.5bit \
    gmm_indexes/gist-960-euclidean_train_reduced.5bit.index \
    gmm_indexes/gist-960-euclidean_train_reduced.5bit.gmm

    ./test_clustering \
    quantised_data/glove-25-angular_train_reduced.5bit \
    gmm_indexes/glove-25-angular_train_reduced.5bit.index \
    gmm_indexes/glove-25-angular_train_reduced.5bit.gmm

    ./test_clustering \
    quantised_data/glove-50-angular_train_reduced.5bit \
    gmm_indexes/glove-50-angular_train_reduced.5bit.index \
    gmm_indexes/glove-50-angular_train_reduced.5bit.gmm

    ./test_clustering \
    quantised_data/glove-100-angular_train_reduced.5bit \
    gmm_indexes/glove-100-angular_train_reduced.5bit.index \
    gmm_indexes/glove-100-angular_train_reduced.5bit.gmm

    ./test_clustering \
    quantised_data/glove-200-angular_train_reduced.5bit \
    gmm_indexes/glove-200-angular_train_reduced.5bit.index \
    gmm_indexes/glove-200-angular_train_reduced.5bit.gmm

    ./test_clustering \
    quantised_data/lastfm-64-dot_train_reduced.5bit \
    gmm_indexes/lastfm-64-dot_train_reduced.5bit.index \
    gmm_indexes/lastfm-64-dot_train_reduced.5bit.gmm

    ./test_clustering \
    quantised_data/mnist-784-euclidean_train_reduced.5bit \
    gmm_indexes/mnist-784-euclidean_train_reduced.5bit.index \
    gmm_indexes/mnist-784-euclidean_train_reduced.5bit.gmm

    ./test_clustering \
    quantised_data/nytimes-256-angular_train_reduced.5bit \
    gmm_indexes/nytimes-256-angular_train_reduced.5bit.index \
    gmm_indexes/nytimes-256-angular_train_reduced.5bit.gmm

    ./test_clustering \
    quantised_data/sift-128-euclidean_train_reduced.5bit \
    gmm_indexes/sift-128-euclidean_train_reduced.5bit.index \
    gmm_indexes/sift-128-euclidean_train_reduced.5bit.gmm

*/

#define MAX_SAMPLE 5000   // limit number of vectors to sample for speed

// Read cluster index file (.index)
int read_index(const char* filename, uint32_t** cluster_ids, uint32_t* n, uint32_t* dim, uint32_t* k) {
    FILE* f = fopen(filename, "rb");
    if (!f) { perror("fopen"); return -1; }

    fread(n, sizeof(uint32_t), 1, f);
    fread(dim, sizeof(uint32_t), 1, f);
    fread(k, sizeof(uint32_t), 1, f);
    *cluster_ids = (uint32_t*)malloc(*n * sizeof(uint32_t));
    fread(*cluster_ids, sizeof(uint32_t), *n, f);
    fclose(f);
    return 0;
}

// Read GMM file (.gmm)
int read_gmm(const char* filename, gmm_params_t* gmm) {
    FILE* f = fopen(filename, "rb");
    if (!f) { perror("fopen"); return -1; }

    fread(&gmm->dim, sizeof(uint32_t), 1, f);
    fread(&gmm->variance, sizeof(float), 1, f);
    fread(gmm->weights, sizeof(float), K, f);
    for (uint32_t k = 0; k < K; k++) {
        fread(gmm->means[k], sizeof(float), gmm->dim, f);
    }
    fclose(f);
    return 0;
}

// Compute L2 distance
float l2_distance(const float* a, const float* b, uint32_t dim) {
    float acc = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        acc += diff * diff;
    }
    return sqrtf(acc);
}

// Simple random number 0..n-1
static inline uint32_t rand_idx(uint32_t n) { return (uint32_t)((rand() / (float)RAND_MAX) * n); }

// Dequantize a packed dataset (reuse from gmm_cluster_runner.c)
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
    if (argc != 4) {
        printf("Usage: %s <dataset.5bit> <dataset.index> <dataset.gmm>\n", argv[0]);
        return 1;
    }

    const char* f_quant = argv[1];
    const char* f_index = argv[2];
    const char* f_gmm   = argv[3];

    printf("=== GMM Cluster Diagnostic ===\n");
    printf("Dataset: %s\n", f_quant);

    // 1️⃣ Read quantised data header
    header_t header;
    uint8_t* packed = read_packed(f_quant, &header);
    if (!packed) { fprintf(stderr, "Error: Cannot read .5bit file\n"); return 1; }
    printf("n=%u, dim=%u, range=[%.4f, %.4f]\n", header.n, header.dim, header.min_val, header.max_val);

    // 2️⃣ Read .index file
    uint32_t *cluster_ids, n_idx, dim_idx, k_idx;
    if (read_index(f_index, &cluster_ids, &n_idx, &dim_idx, &k_idx)) {
        fprintf(stderr, "Failed to read index file\n"); return 1;
    }

    if (n_idx != header.n || dim_idx != header.dim) {
        fprintf(stderr, "❌ Mismatch between quantised file and index header!\n");
        return 1;
    }
    printf("Index file: %u vectors, %u dims, K=%u\n", n_idx, dim_idx, k_idx);

    // 3️⃣ Read .gmm parameters
    gmm_params_t gmm;
    if (read_gmm(f_gmm, &gmm)) {
        fprintf(stderr, "Failed to read gmm file\n"); return 1;
    }
    printf("GMM: dim=%u, variance=%.4f\n", gmm.dim, gmm.variance);

    // Check weight sum
    float wsum = 0.0f;
    for (uint32_t i = 0; i < K; i++) wsum += gmm.weights[i];
    printf("Weights: ");
    for (uint32_t i = 0; i < K; i++) printf("%.3f ", gmm.weights[i]);
    printf("\nSum(weights)=%.6f %s\n", wsum, fabsf(wsum - 1.0f) < 1e-3 ? "✅" : "⚠️");

    // 4️⃣ Check cluster balance
    uint32_t* counts = (uint32_t*)calloc(K, sizeof(uint32_t));
    for (uint32_t i = 0; i < n_idx; i++) {
        if (cluster_ids[i] < K) counts[cluster_ids[i]]++;
    }
    printf("\nCluster distribution:\n");
    for (uint32_t k = 0; k < K; k++) {
        float pct = (counts[k] * 100.0f) / n_idx;
        printf("  Cluster %u: %u (%.2f%%)\n", k, counts[k], pct);
    }

    // 5️⃣ Probabilities sanity check (sum ≈ 1)
    srand(42);
    float** data = dequantize_all(packed, &header);
    uint32_t test_id = rand_idx(header.n);
    float* q = data[test_id];
    float probs[K];
    compute_cluster_probabilities(q, &gmm, probs);
    float psum = 0.0f;
    for (uint32_t k = 0; k < K; k++) psum += probs[k];
    printf("\nRandom sample (vector %u) prob sum = %.6f %s\n", test_id, psum, fabsf(psum - 1.0f) < 1e-5 ? "✅" : "⚠️");

    // 6️⃣ Optional: within-cluster variance estimate (small sample)
    printf("\nWithin-cluster distance summary (sample ≤ %d per cluster):\n", MAX_SAMPLE);
    for (uint32_t k = 0; k < K; k++) {
        uint32_t sample_count = 0;
        double mean_dist = 0.0;
        for (uint32_t i = 0; i < header.n && sample_count < MAX_SAMPLE; i++) {
            if (cluster_ids[i] == k) {
                float d = l2_distance(data[i], gmm.means[k], header.dim);
                mean_dist += d;
                sample_count++;
            }
        }
        if (sample_count > 0) mean_dist /= sample_count;
        printf("  Cluster %u: mean distance = %.4f (n=%u)\n", k, (float)mean_dist, sample_count);
    }

    printf("\n✅ GMM diagnostic complete.\n\n");

    free(cluster_ids);
    free(counts);
    for (uint32_t i = 0; i < header.n; i++) free(data[i]);
    free(data);
    free(packed);
    return 0;
}
