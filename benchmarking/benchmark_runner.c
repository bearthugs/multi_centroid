#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "faiss_wrapper.h"
#include "benchmark_timer.h"
#include "benchmark_utils.h"

// Declare load_fvecs
float* load_fvecs(const char* filename, int* out_num_vectors, int* out_dim);

void run_benchmark(const char* base_vectors_file, const char* query_vectors_file, int k) {
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
    int* gt = load_ivecs("../fvecs_data/sift-128-euclidean_neighbours.ivecs", &gt_nq, &gt_k);
    if (!gt || gt_nq != n_query) {
        fprintf(stderr, "Ground truth size mismatch\n");
        return;
    }

    mkdir("results", 0777);

    FILE* csv = fopen("./results/benchmark_results.csv", "w");
    if (!csv) {
        perror("Failed to open CSV file");
        return;
    }

    fprintf(csv, "target_recall,efSearch,recall,query_time\n");

    // 1. Index build (construction overhead)
    printf("Building FAISS HNSW index...\n");
    double t0 = wall_time();
    FaissIndex index = faiss_create_hnsw_index(d_base, 32);
    faiss_add_vectors(index, base_vectors, n_base, d_base);
    double t_build = wall_time() - t0;
    printf("Construction time: %.3f s\n", t_build);

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

    // 4. Accuracy (recall)
    double recall = compute_recall_at_k(labels, n_query, k, gt, gt_k);
    printf("Recall@%d = %.4f\n", k, recall);

    // 5. Sweep for target recalls
    double targets[] = {0.95, 0.98, 0.99};
    for (int ti = 0; ti < 3; ti++) {
        double target = targets[ti];
        printf("\n--- Benchmark for target recall %.2f ---\n", target);

        for (int ef = 10; ef <= 500; ef += 10) {
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
