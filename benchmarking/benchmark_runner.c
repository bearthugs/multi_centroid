#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "faiss_wrapper.h"

// Declare load_fvecs
float* load_fvecs(const char* filename, int* out_num_vectors, int* out_dim);

void run_benchmark(const char* base_vectors_file, const char* query_vectors_file, int k) {
    int d_base, n_base;
    float* base_vectors = load_fvecs(base_vectors_file, &n_base, &d_base);
    if (!base_vectors) {
        fprintf(stderr, "Failed to load base vectors\n");
        return;
    }

    int d_query, n_query;
    float* query_vectors = load_fvecs(query_vectors_file, &n_query, &d_query);
    if (!query_vectors) {
        fprintf(stderr, "Failed to load query vectors\n");
        free(base_vectors);
        return;
    }

    if (d_base != d_query) {
        fprintf(stderr, "Dimension mismatch between base and query vectors\n");
        free(base_vectors);
        free(query_vectors);
        return;
    }

    printf("Building FAISS HNSW index with %d vectors, dimension %d...\n", n_base, d_base);
    FaissIndex index = faiss_create_hnsw_index(d_base, 32);
    faiss_add_vectors(index, base_vectors, n_base, d_base);

    printf("Searching %d queries for %d nearest neighbors each...\n", n_query, k);
    float* distances = (float*)malloc(n_query * k * sizeof(float));
    //int64_t* labels = (int64_t*)malloc(n_query * k * sizeof(int64_t));
    int64_t* labels = malloc(n_query * k * sizeof(int64_t));
    if (!distances || !labels) {
        fprintf(stderr, "Failed to allocate search output arrays\n");
        faiss_free_index(index);
        free(base_vectors);
        free(query_vectors);
        free(distances);
        free(labels);
        return;
    }

    faiss_search(index, n_query, query_vectors, d_query, k, distances, labels);

    // Print first 5 results for the first query as example
    printf("Top %d results for first query:\n", k);
    for (int i = 0; i < k; i++) {
        printf("  rank %d: id=%lld dist=%f\n", i, labels[i], distances[i]);
    }

    // Cleanup
    faiss_free_index(index);
    free(base_vectors);
    free(query_vectors);
    free(distances);
    free(labels);
}
