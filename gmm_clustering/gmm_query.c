#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "utils.h"
#include "../5bit_quantisation/quant_functions.h"
#include "gmm_cluster_runner.h"

#define DEFAULT_KNN 100
#define QUERY_THRESH 0.0f   // t_q: min prob for query clusters
#define EDGE_THRESH  0.0f   // t_e: min prob for neighbour clusters

// gcc -O2 gmm_clustering/gmm_query.c gmm_clustering/utils.c quant_functions.c -lm -o gmm_query

// ./gmm_query fvecs_data/sift-128-euclidean_test.fvecs \
    reduced_data/sift-128-euclidean_train_svd_model.bin \
    gmm_indexes/sift-128-euclidean_train_reduced.5bit.gmm \
    gmm_indexes/sift-128-euclidean_train_reduced.5bit.index \
    100 > results/sift-128_retrieved.txt

typedef struct {
    uint32_t n;
    uint32_t dim;
    uint32_t *assignments;  // cluster id per vector
    header_t header;
    uint8_t *packed;        // quantised vectors
} index_t;

static index_t* load_index(const char *index_path, const char *data_path) {
    FILE *f = fopen(index_path, "rb");
    if (!f) { perror("open index"); return NULL; }

    index_t *idx = calloc(1, sizeof(index_t));
    fread(&idx->n, sizeof(uint32_t), 1, f);
    fread(&idx->dim, sizeof(uint32_t), 1, f);
    uint32_t k_val; fread(&k_val, sizeof(uint32_t), 1, f);
    idx->assignments = malloc(idx->n * sizeof(uint32_t));
    fread(idx->assignments, sizeof(uint32_t), idx->n, f);
    fclose(f);

    // load packed 5-bit data
    idx->packed = read_packed(data_path, &idx->header);
    return idx;
}

static void free_index(index_t *idx) {
    if (!idx) return;
    free(idx->assignments);
    free(idx->packed);
    free(idx);
}

// find KNN within pool
static void knn_search(const index_t *idx, const float *query,
                       const gmm_params_t *gmm,
                       const uint32_t *pool, uint32_t pool_size, uint32_t k)
{
    float *dist = malloc(pool_size * sizeof(float));
    uint32_t *ids = malloc(pool_size * sizeof(uint32_t));

    const uint32_t dim = idx->header.dim;
    const float min_val = idx->header.min_val;
    const float max_val = idx->header.max_val;
    const float range = fmaxf(max_val - min_val, 1e-9f);

    float *vec = malloc(dim * sizeof(float));

    for (uint32_t i = 0; i < pool_size; i++) {
        ids[i] = pool[i];

        // --- unpack dataset vector from packed representation ---
        uint32_t start_bit = pool[i] * dim * BIT_DEPTH;
        for (uint32_t d = 0; d < dim; d++) {
            uint32_t bit_pos = start_bit + d * BIT_DEPTH;
            uint8_t qval = 0;
            for (uint8_t b = 0; b < BIT_DEPTH; b++) {
                uint32_t byte_idx = (bit_pos + b) / 8;
                uint8_t bit_offset = (bit_pos + b) % 8;
                if (idx->packed[byte_idx] & (1 << bit_offset))
                    qval |= (1 << b);
            }
            vec[d] = min_val + (qval / (float)(LEVELS - 1)) * range;
        }

        // --- compute Euclidean distance to query ---
        float acc = 0.0f;
        for (uint32_t d = 0; d < dim; d++) {
            float diff = query[d] - vec[d];
            acc += diff * diff;
        }
        dist[i] = sqrtf(acc);
    }

    free(vec);

    // partial selection sort for top-k
    for (uint32_t i = 0; i < k && i < pool_size; i++) {
        uint32_t best = i;
        for (uint32_t j = i + 1; j < pool_size; j++)
            if (dist[j] < dist[best]) best = j;
        float td = dist[i]; dist[i] = dist[best]; dist[best] = td;
        uint32_t ti = ids[i]; ids[i] = ids[best]; ids[best] = ti;
    }

    // === OUTPUT: validation-friendly (only IDs) ===
    for (uint32_t i = 0; i < k && i < pool_size; i++) {
        printf("%u", ids[i]);
        if (i < k - 1) printf(" ");
        else printf("\n");
    }

    free(dist);
    free(ids);
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr,
            "Usage: %s <query.fvecs> <svd_model.bin> <gmm_file.gmm> <index_file.index> [topK]\n",
            argv[0]);
        return 1;
    }
    const char *query_path = argv[1];
    const char *svd_path   = argv[2];
    const char *gmm_path   = argv[3];
    const char *index_path = argv[4];
    uint32_t KNN = (argc > 5) ? atoi(argv[5]) : DEFAULT_KNN;

    // --- load test vectors ---
    uint32_t n_query, orig_dim;
    float **queries = read_fvecs_2d(query_path, &n_query, &orig_dim);
    if (!queries) return 1;

    // --- load GMM params ---
    FILE *fg = fopen(gmm_path, "rb");
    if (!fg) { perror("open gmm"); return 1; }
    gmm_params_t gmm;
    fread(&gmm.dim, sizeof(uint32_t), 1, fg);
    fread(&gmm.variance, sizeof(float), 1, fg);
    fread(gmm.weights, sizeof(float), K, fg);
    for (uint32_t k=0;k<K;k++) fread(gmm.means[k], sizeof(float), gmm.dim, fg);
    fclose(fg);

    // --- load index + packed data ---
    const char *data_path = "quantised_data/sift-128-euclidean_train_reduced.5bit"; // adjust per dataset
    index_t *idx = load_index(index_path, data_path);
    if (!idx) return 1;

    fprintf(stderr, "Loaded index with %u vectors, %u dims.\n", idx->n, idx->dim);

    // --- process each query ---
    for (uint32_t qi = 0; qi < n_query; qi++) {
        float *reduced = reduce_vector_with_svd(queries[qi], orig_dim, svd_path);
        if (!reduced) continue;

        float probs[K]; compute_cluster_probabilities(reduced, &gmm, probs);

        uint32_t cluster_flags[K] = {0};
        uint32_t *pool = malloc(idx->n * sizeof(uint32_t));
        uint32_t pool_size = 0;

        for (uint32_t c=0;c<K;c++) {
            if (probs[c] >= QUERY_THRESH) {
                cluster_flags[c] = 1;
                for (uint32_t i=0;i<idx->n;i++)
                    if (idx->assignments[i]==c)
                        pool[pool_size++] = i;
            }
        }

        // expand pool using EDGE_THRESH
        for (uint32_t i=0;i<pool_size;i++) {
            uint32_t pid = pool[i];
            uint32_t c = idx->assignments[pid];
            float pc[K]; compute_cluster_probabilities(gmm.means[c], &gmm, pc);
            for (uint32_t j=0;j<K;j++) {
                if (!cluster_flags[j] && pc[j] >= EDGE_THRESH) {
                    cluster_flags[j] = 1;
                    for (uint32_t t=0;t<idx->n;t++)
                        if (idx->assignments[t]==j)
                            pool[pool_size++] = t;
                }
            }
        }

        fprintf(stderr, "Query %u: pool_size=%u\n", qi, pool_size);
        knn_search(idx, reduced, &gmm, pool, pool_size, KNN);
        free(pool);
        free(reduced);
    }

    free_fvecs(queries, n_query);
    free_index(idx);
    return 0;
}
