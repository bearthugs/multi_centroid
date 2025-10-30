#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <string.h>
#include <math.h>
#include "faiss_wrapper.h"
#include "benchmark_timer.h"
#include "benchmark_utils.h"

// Declare load_fvecs
float* load_fvecs(const char* filename, int* out_num_vectors, int* out_dim);

void get_dataset_name(const char* query_path, char* dataset_name, size_t size) {
    // Extract just the filename part
    const char* filename = strrchr(query_path, '/');
    if (!filename) {
        filename = query_path;
    } else {
        filename++; // skip '/'
    }

    // Copy filename into buffer
    strncpy(dataset_name, filename, size - 1);
    dataset_name[size - 1] = '\0';

    // Remove the "_test.fvecs" suffix
    char* suffix = strstr(dataset_name, "_test.fvecs");
    if (suffix) {
        *suffix = '\0';
    }
}

void normalize_vectors(float* data, int n, int d) {
    for (int i = 0; i < n; i++) {
        float norm = 0.0f;
        for (int j = 0; j < d; j++)
            norm += data[i * d + j] * data[i * d + j];
        norm = sqrtf(norm);
        if (norm > 1e-9f) { // small epsilon to avoid div by zero
            for (int j = 0; j < d; j++)
                data[i * d + j] /= norm;
        }
    }
}

void run_benchmark_hnsw(const char* base_vectors_file, const char* query_vectors_file, const char* neighbours_file, int k, int metric) {
    struct stat st;
    FaissIndex index = NULL;
    char index_filename[256];
    char dataset_name[256];

    double t0 = 0;
    double t_build = 0;

    int M = 64;
    int efConstruction = 100;

    int d_base, n_base;
    float* base_vectors = load_fvecs(base_vectors_file, &n_base, &d_base);

    int d_query, n_query;
    float* query_vectors = load_fvecs(query_vectors_file, &n_query, &d_query);

    if (d_base != d_query) {
        fprintf(stderr, "Dimension mismatch\n");
        return;
    }

    // Load ground truth
    int gt_nq, gt_k;
    int* gt = load_ivecs(neighbours_file, &gt_nq, &gt_k);
    if (!gt || gt_nq != n_query) {
        fprintf(stderr, "Ground truth size mismatch\n");
        return;
    }

    mkdir("sorted_results/results_M64_efC100", 0777);

    FILE* csv = fopen("./sorted_results/results_M64_efC100/benchmark_results.csv", "w");
    if (!csv) {
        perror("Failed to open CSV file");
        return;
    }

    //fprintf(csv, "dataset,construction_time(ms),query_time(ms),recall@100,M,efConstruction\n");

    get_dataset_name(query_vectors_file, dataset_name, sizeof(dataset_name));

    snprintf(index_filename, sizeof(index_filename), "./sorted_results/results_M64_efC100/%s_hnsw.index", dataset_name);

    // 1. Index build (construction overhead)
    if (stat(index_filename, &st) == 0) {
        printf("Loading existing HNSW index...\n");
        index = faiss_load_index(index_filename);
    }
    else {
        printf("Building FAISS HNSW index...\n");
        t0 = wall_time();
        if (metric == 1) {
            normalize_vectors(base_vectors, n_base, d_base);
            normalize_vectors(query_vectors, n_query, d_query);
        }
        index = faiss_create_hnsw_index(d_base, M, efConstruction, metric); // metric = 1 for cosine
        faiss_add_vectors(index, base_vectors, n_base, d_base);
        t_build = wall_time() - t0;
        printf("Construction time: %.3f s\n", t_build);
    }

    // 2. Search (query overhead)
    float* distances = malloc(n_query * k * sizeof(float));
    int64_t* labels = malloc(n_query * k * sizeof(int64_t));

    printf("Searching...\n");
    t0 = wall_time();
    faiss_search(index, n_query, query_vectors, d_query, k, distances, labels);
    double t_search = wall_time() - t0;
    printf("Query time: %.3f s\n", t_search);

    // 3. Combined
    printf("Total time (build + search): %.3f s\n", t_build + t_search);

    // Save the constructed HNSW graph
    printf("Saving HNSW index...\n");
    faiss_save_index(index, index_filename);

    // 4. Accuracy (recall)
    double recall = compute_recall_at_k(labels, n_query, k, gt, gt_k);
    printf("Recall@%d = %.4f\n", k, recall);

    // Save to .csv file
    fprintf(csv, "%s,%.3f,%.3f,%.4f,%d,%d\n",
            dataset_name,
            t_build * 1000.0,
            t_search * 1000.0,
            recall,
            M,
            efConstruction);
    
    
    // 5. Sweep for target recalls
    double targets[] = {0.99};
    for (int ti = 0; ti < 1; ti++) {
        double target = targets[ti];
        printf("\n--- Benchmark for target recall %.2f ---\n", target);

        for (int ef = 200; ef <= 200; ef += 1) {
            faiss_hnsw_set_efSearch(index, ef);

            double t0 = wall_time();
            faiss_search(index, n_query, query_vectors, d_query, k, distances, labels);
            double t_search = wall_time() - t0;

            double recall = compute_recall_at_k(labels, n_query, k, gt, gt_k);

            printf("efSearch=%d  recall=%.4f  query_time=%.3f s\n", ef, recall, t_search);

            // Save to CSV
            fprintf(csv, "%.2f,%d,%.4f,%.6f\n", target, ef, recall, t_search);

            if (recall >= target) {
                printf("Reached target recall %.2f with efSearch=%d\n", target, ef);
                break;
            }
        }
    }
    
    free(base_vectors);
    free(query_vectors);
    free(distances);
    free(labels);
    free(gt);
    faiss_free_index(index);
}

void run_benchmark_ivf(const char* base_vectors_file, const char* query_vectors_file, const char* neighbours_file, int k) {

}