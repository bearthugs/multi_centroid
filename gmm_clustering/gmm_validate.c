#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "utils.h"
#include "../quant_functions.h"

/*
COMPILING:
    gcc -O2 gmm_clustering/gmm_validate.c gmm_clustering/utils.c quant_functions.c -lm -o gmm_validate

RUNNING:
    ./gmm_validate results/coco-i2i-512_retrieved.txt coco-i2i-512-angular
    ./gmm_validate results/sift-128_retrieved.txt sift-128-euclidean
*/

// Load neighbours (.ivecs)
int** read_ivecs(const char* filename, uint32_t* out_n, uint32_t* out_k) {
    FILE* f = fopen(filename, "rb");
    if (!f) { perror("fopen ivecs"); return NULL; }

    // find size and dimension
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    rewind(f);

    uint32_t k;
    fread(&k, sizeof(uint32_t), 1, f);
    size_t entry_size = sizeof(uint32_t) + k * sizeof(int32_t);
    *out_n = fsize / entry_size;
    *out_k = k;

    int** data = malloc(*out_n * sizeof(int*));
    rewind(f);
    for (uint32_t i = 0; i < *out_n; i++) {
        uint32_t d;
        fread(&d, sizeof(uint32_t), 1, f);
        data[i] = malloc(d * sizeof(int));
        fread(data[i], sizeof(int), d, f);
    }
    fclose(f);
    return data;
}

// Recall@K calculation
float compute_recall_at_k(uint32_t** retrieved, uint32_t n_query,
                          uint32_t k_retrieved, int** gt_neighbours,
                          uint32_t gt_k)
{
    uint32_t correct = 0;
    for (uint32_t i = 0; i < n_query; i++) {
        for (uint32_t r = 0; r < k_retrieved; r++) {
            for (uint32_t g = 0; g < gt_k; g++) {
                if (retrieved[i][r] == (uint32_t)gt_neighbours[i][g]) {
                    correct++;
                    break;
                }
            }
        }
    }
    float denom = (float)(n_query * k_retrieved);
    return correct / denom;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <retrieved_results.txt> <dataset_name>\n", argv[0]);
        return 1;
    }

    const char* retrieved_path = argv[1];
    const char* dataset_name   = argv[2];

    // ================================
    // Load ground-truth .ivecs & .fvecs
    // ================================
    char gt_nei_path[512], gt_dist_path[512];
    snprintf(gt_nei_path, sizeof(gt_nei_path), "fvecs_data/%s_neighbours.ivecs", dataset_name);
    snprintf(gt_dist_path, sizeof(gt_dist_path), "fvecs_data/%s_distances.fvecs", dataset_name);

    uint32_t n_gt, gt_k;
    int** gt_nei = read_ivecs(gt_nei_path, &n_gt, &gt_k);
    if (!gt_nei) return 1;

    uint32_t n_dist, dim;
    float* gt_dist = read_fvecs(gt_dist_path, &n_dist, &dim);
    free(gt_dist); // only for dimension check

    // ================================
    // Load retrieved result list
    // Format: each line contains top-k IDs
    // ================================
    FILE* f = fopen(retrieved_path, "r");
    if (!f) { perror("fopen retrieved"); return 1; }

    uint32_t n_query = n_gt;
    uint32_t k_retrieved = 100;
    uint32_t** retrieved = malloc(n_query * sizeof(uint32_t*));
    for (uint32_t i = 0; i < n_query; i++) {
        retrieved[i] = malloc(k_retrieved * sizeof(uint32_t));
        for (uint32_t j = 0; j < k_retrieved; j++) {
            if (fscanf(f, "%u", &retrieved[i][j]) != 1)
                retrieved[i][j] = 0;
        }
    }
    fclose(f);

    // ================================
    // Compute Recall@100
    // ================================
    float recall = compute_recall_at_k(retrieved, n_query, k_retrieved, gt_nei, gt_k);
    printf("Recall@%u = %.4f\n", k_retrieved, recall);

    for (uint32_t i = 0; i < n_query; i++) free(retrieved[i]);
    free(retrieved);
    for (uint32_t i = 0; i < n_gt; i++) free(gt_nei[i]);
    free(gt_nei);
    return 0;
}
