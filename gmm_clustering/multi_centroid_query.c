#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "../quant_functions.h"
#include "utils.h"  // reduce_vector_with_svd, quantise_vector_5bit, get_top_clusters

#define TOP_K_NEIGHBORS 100
#define TOP_CLUSTERS 3

/*
COMPILING:
    gcc -O2 gmm_clustering/multi_centroid_query.c gmm_clustering/utils.c quant_functions.c -lm -o multi_centroid_query

./multi_centroid_query fvecs_data/coco-i2i-512-angular_test.fvecs reduced_data/coco-i2i-512-angular_train_svd_model.pkl gmm_indexes/coco-i2i-512-angular_train_reduced.gmm quantised_data/coco-i2i-512-angular_train_reduced.5bit gmm_indexes/coco-i2i-512-angular_train_reduced.5bit.index

*/

int main(int argc, char** argv) {
    if (argc < 6) {
        printf("Usage: %s <query_fvecs> <svd_model> <gmm_file> <packed_dataset> <cluster_index>\n", argv[0]);
        return 1;
    }

    const char* query_file = argv[1];
    const char* svd_model = argv[2];
    const char* gmm_file = argv[3];
    const char* packed_file = argv[4];
    const char* cluster_index_file = argv[5];

    // -----------------------
    // Load queries
    // -----------------------
    uint32_t n_queries, dim_queries;
    float** queries = read_fvecs_2d(query_file, &n_queries, &dim_queries);
    if (!queries) return 1;
    printf("Loaded %u queries of dimension %u\n", n_queries, dim_queries);

    // -----------------------
    // Load packed dataset
    // -----------------------
    header_t dataset_header;
    uint8_t* packed_data = read_packed(packed_file, &dataset_header);
    if (!packed_data) {
        free(queries);
        return 1;
    }

    // -----------------------
    // Load cluster assignments
    // -----------------------
    FILE* f = fopen(cluster_index_file, "rb");
    if (!f) {
        perror("Failed to open cluster index file");
        free(queries);
        free(packed_data);
        return 1;
    }

    uint32_t n_vectors, dim_index, K_clusters;
    fread(&n_vectors, sizeof(uint32_t), 1, f);
    fread(&dim_index, sizeof(uint32_t), 1, f);
    fread(&K_clusters, sizeof(uint32_t), 1, f);

    if (n_vectors != dataset_header.n) {
        fprintf(stderr, "Cluster index count (%u) does not match dataset (%u)\n", n_vectors, dataset_header.n);
        fclose(f);
        free(queries);
        free(packed_data);
        return 1;
    }

    uint32_t* cluster_ids = (uint32_t*)malloc(n_vectors * sizeof(uint32_t));
    fread(cluster_ids, sizeof(uint32_t), n_vectors, f);
    fclose(f);

    // -----------------------
    // Precompute vectors per cluster
    // -----------------------
    uint32_t* cluster_counts = (uint32_t*)calloc(K_clusters, sizeof(uint32_t));
    for (uint32_t i = 0; i < n_vectors; i++)
        cluster_counts[cluster_ids[i]]++;

    uint32_t** cluster_members = (uint32_t**)malloc(K_clusters * sizeof(uint32_t*));
    for (uint32_t k = 0; k < K_clusters; k++)
        cluster_members[k] = (uint32_t*)malloc(cluster_counts[k] * sizeof(uint32_t));

    // Fill members
    uint32_t* filled = (uint32_t*)calloc(K_clusters, sizeof(uint32_t));
    for (uint32_t i = 0; i < n_vectors; i++) {
        uint32_t c = cluster_ids[i];
        cluster_members[c][filled[c]++] = i;
    }
    free(filled);

    // -----------------------
    // Process queries
    // -----------------------
    for (uint32_t i = 0; i < n_queries; i++) {
        float* qvec = queries[i];

        // Reduce
        float* reduced = reduce_vector_with_svd(qvec, dim_queries, svd_model);
        if (!reduced) continue;

        // Top clusters
        uint32_t* top_clusters = get_top_clusters(reduced, dataset_header.dim,
                                                  dataset_header.min_val,
                                                  dataset_header.max_val,
                                                  gmm_file,
                                                  TOP_CLUSTERS);
        if (!top_clusters) {
            free(reduced);
            continue;
        }

        float best_distances[TOP_K_NEIGHBORS];
        uint32_t best_indices[TOP_K_NEIGHBORS];
        for (uint32_t k = 0; k < TOP_K_NEIGHBORS; k++) {
            best_distances[k] = 1e9;
            best_indices[k] = UINT32_MAX;
        }

        // Search only vectors in top clusters
        for (uint32_t c = 0; c < TOP_CLUSTERS; c++) {
            uint32_t cluster_id = top_clusters[c];
            for (uint32_t j = 0; j < cluster_counts[cluster_id]; j++) {
                uint32_t idx = cluster_members[cluster_id][j];
                float dist = distance(packed_data, idx, i, &dataset_header, DIST_EUCLIDEAN);

                // Insert into sorted top-K
                for (uint32_t k = 0; k < TOP_K_NEIGHBORS; k++) {
                    if (dist < best_distances[k]) {
                        for (uint32_t l = TOP_K_NEIGHBORS - 1; l > k; l--) {
                            best_distances[l] = best_distances[l - 1];
                            best_indices[l] = best_indices[l - 1];
                        }
                        best_distances[k] = dist;
                        best_indices[k] = idx;
                        break;
                    }
                }
            }
        }

        // Output
        printf("Query %u top-%u neighbors:\n", i, TOP_K_NEIGHBORS);
        for (uint32_t k = 0; k < TOP_K_NEIGHBORS; k++) {
            printf("%u (%.4f) ", best_indices[k], best_distances[k]);
        }
        printf("\n");

        free(reduced);
        free(top_clusters);
    }

    // -----------------------
    // Cleanup
    // -----------------------
    free(queries);
    free(packed_data);
    free(cluster_ids);
    free(cluster_counts);
    for (uint32_t k = 0; k < K_clusters; k++) free(cluster_members[k]);
    free(cluster_members);

    return 0;
}
