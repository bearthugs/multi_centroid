// gmm_query.c
// ------------------------------------------------------------
// Query-time ANN candidate generation using GMM cluster expansion.
//
// INPUTS:
//   1) train .5bit file      (quantised training vectors)
//   2) test  .5bit file      (quantised query vectors)
//   3) .gmm file             (trained mixture over training space)
//   4) cluster_to_vectors index for the *training* set
//   5) t_query (float)       threshold for query's cluster posteriors
//   6) t_edge  (float)       threshold for pool-vertex cluster posteriors
//
// OUTPUT:
//   querying_results/<test_basename>_gmm_candidates.ivecs
//
//   Each row is FAISS-style ivecs:
//       [int32 100] [int32 id0] ... [int32 id99]
//   where ids are training vector indices.
//
// COMPILE:
//   gcc -O2 gmm_querying/gmm_query.c 5bit_quantisation/quant_functions.c -lm -o gmm_query
//
/*
    RUN:
        ./gmm_query \
            quantised_data/train/coco-i2i-512-angular_train_reduced.5bit \
            quantised_data/test/coco-i2i-512-angular_test_reduced.5bit \
            gmm_indexes/gmm/K8/coco-i2i-512-angular_train_reduced.gmm \
            gmm_indexes/cluster_to_vectors/K8/coco-i2i-512-angular_train_reduced.index \
            0.8 0.5

        ./gmm_query \
            quantised_data/train/coco-i2i-512-angular_train_reduced.5bit \
            quantised_data/test/coco-i2i-512-angular_test_reduced.5bit \
            gmm_indexes/gmm/K16/coco-i2i-512-angular_train_reduced.gmm \
            gmm_indexes/cluster_to_vectors/K16/coco-i2i-512-angular_train_reduced.index \
            0.8 0.5

        ./gmm_query \
            quantised_data/train/coco-i2i-512-angular_train_reduced.5bit \
            quantised_data/test/coco-i2i-512-angular_test_reduced.5bit \
            gmm_indexes/gmm/K32/coco-i2i-512-angular_train_reduced.gmm \
            gmm_indexes/cluster_to_vectors/K32/coco-i2i-512-angular_train_reduced.index \
            0.8 0.5

        ./gmm_query \
            quantised_data/train/coco-i2i-512-angular_train_reduced.5bit \
            quantised_data/test/coco-i2i-512-angular_test_reduced.5bit \
            gmm_indexes/gmm/K64/coco-i2i-512-angular_train_reduced.gmm \
            gmm_indexes/cluster_to_vectors/K64/coco-i2i-512-angular_train_reduced.index \
            0.8 0.5

        ./gmm_query \
            quantised_data/train/coco-i2i-512-angular_train_reduced.5bit \
            quantised_data/test/coco-i2i-512-angular_test_reduced.5bit \
            gmm_indexes/gmm/K128/coco-i2i-512-angular_train_reduced.gmm \
            gmm_indexes/cluster_to_vectors/K128/coco-i2i-512-angular_train_reduced.index \
            0.7 0.5

        ./gmm_query \
            quantised_data/train/gist-960-euclidean_train_reduced.5bit \
            quantised_data/test/gist-960-euclidean_test_reduced.5bit \
            gmm_indexes/gmm/K128/gist-960-euclidean_train_reduced.gmm \
            gmm_indexes/cluster_to_vectors/K128/gist-960-euclidean_train_reduced.index \
            0.8 0.5

        ./gmm_query \
            quantised_data/train/glove-50-angular_train_reduced.5bit \
            quantised_data/test/glove-50-angular_test_reduced.5bit \
            gmm_indexes/gmm/K128/glove-50-angular_train_reduced.gmm \
            gmm_indexes/cluster_to_vectors/K128/glove-50-angular_train_reduced.index \
            0.8 0.5

        ./gmm_query \
            quantised_data/train/glove-25-angular_train_reduced.5bit \
            quantised_data/test/glove-25-angular_test_reduced.5bit \
            gmm_indexes/gmm/K128/glove-25-angular_train_reduced.gmm \
            gmm_indexes/cluster_to_vectors/K128/glove-25-angular_train_reduced.index \
            0.8 0.5
*/
// ------------------------------------------------------------




#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <time.h>


#ifdef _WIN32
  #include <direct.h>
  #define MKDIR(p) _mkdir(p)
#else
  #include <sys/stat.h>
  #include <sys/types.h>
  #define MKDIR(p) mkdir(p, 0755)
#endif

#include "../5bit_quantisation/quant_functions.h"  // adjust path if needed

// ====================== GMM header ==========================

typedef struct {
    uint32_t K;
    uint32_t dim;
    uint32_t metric;     // 0=EUCLIDEAN, 1=COSINE
    uint32_t normalised; // 0/1
    float    var_floor;
} gmm_header_t;

// ====================== Small helpers =======================

static void ensure_dir(const char* path) {
    if (MKDIR(path) != 0) {
        (void)errno; // ignore "already exists"
    }
}

static const char* base_name(const char* path) {
    const char* slash1 = strrchr(path, '/');
    const char* slash2 = strrchr(path, '\\');
    const char* s = slash1 ? slash1 + 1 : path;
    if (slash2 && slash2 > slash1) s = slash2 + 1;
    return s;
}

static void strip_ext(const char* name, char* out, size_t outsz) {
    size_t n = strnlen(name, outsz ? outsz-1 : 0);
    size_t i = n;
    while (i > 0 && name[i-1] != '.') i--;
    if (i == 0) {
        snprintf(out, outsz, "%s", name);
    } else {
        size_t base_len = i-1;
        if (base_len >= outsz) base_len = outsz-1;
        memcpy(out, name, base_len);
        out[base_len] = '\0';
    }
}

static inline void l2_normalise(float* v, uint32_t dim) {
    double nrm = 0.0;
    for (uint32_t d=0; d<dim; ++d) {
        nrm += (double)v[d]*(double)v[d];
    }
    nrm = sqrt(nrm);
    if (nrm > 0.0) {
        float inv = (float)(1.0 / nrm);
        for (uint32_t d=0; d<dim; ++d) {
            v[d] *= inv;
        }
    }
}

// unpack_dequantise_vector copied from gmm_clustering.c
static void unpack_dequantise_vector(const uint8_t* packed,
                                     const header_t* h,
                                     uint32_t idx,
                                     float* out) {
    const uint32_t dim   = h->dim;
    const float    minv  = h->min_val;
    const float    range = (h->max_val - h->min_val);
    const uint32_t start_bit = idx * dim * BIT_DEPTH;

    if (range == 0.0f) {
        for (uint32_t d=0; d<dim; ++d) out[d] = minv;
        return;
    }

    for (uint32_t d = 0; d < dim; ++d) {
        uint32_t bit_pos = start_bit + d * BIT_DEPTH;
        uint8_t q = 0;
        for (uint8_t b = 0; b < BIT_DEPTH; ++b) {
            uint32_t byte_idx = (bit_pos + b) / 8;
            uint8_t  bit_off  = (bit_pos + b) % 8;
            if (packed[byte_idx] & (1u << bit_off)) q |= (1u << b);
        }
        out[d] = minv + (range * q) / (LEVELS - 1);
    }
}

static inline float safe_log_sum_exp(const float* arr, uint32_t K) {
    float maxv = arr[0];
    for (uint32_t k=1; k<K; ++k)
        if (arr[k] > maxv) maxv = arr[k];

    double sum = 0.0;
    for (uint32_t k=0; k<K; ++k)
        sum += exp((double)arr[k] - (double)maxv);

    return (float)(log(sum) + (double)maxv);
}

// ====================== GMM loader ==========================

static int load_gmm(const char* path,
                    gmm_header_t* gh,
                    float** weights,
                    float** means,
                    float** variances,
                    float** log_consts) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open GMM file: %s\n", path);
        return -1;
    }

    if (fread(gh, sizeof(gmm_header_t), 1, f) != 1) {
        fprintf(stderr, "Failed to read GMM header\n");
        fclose(f);
        return -1;
    }

    uint32_t K   = gh->K;
    uint32_t dim = gh->dim;

    *weights    = (float*)malloc(sizeof(float) * K);
    *means      = (float*)malloc(sizeof(float) * K * dim);
    *variances  = (float*)malloc(sizeof(float) * K * dim);
    *log_consts = (float*)malloc(sizeof(float) * K);

    if (!*weights || !*means || !*variances || !*log_consts) {
        fprintf(stderr, "OOM when allocating GMM arrays\n");
        fclose(f);
        return -1;
    }

    fread(*weights,    sizeof(float), K,        f);
    fread(*means,      sizeof(float), K*dim,    f);
    fread(*variances,  sizeof(float), K*dim,    f);
    fread(*log_consts, sizeof(float), K,        f);

    fclose(f);
    return 0;
}

// ====================== Posterior P(C|x) ====================

static void posterior_probabilities(
    const float* x,
    const gmm_header_t* gh,
    const float* weights,
    const float* means,
    const float* variances,
    const float* log_consts,
    float* post // out, size K
) {
    uint32_t K   = gh->K;
    uint32_t dim = gh->dim;

    float* logp = (float*)malloc(sizeof(float)*K);
    if (!logp) {
        fprintf(stderr, "OOM in posterior_probabilities\n");
        exit(1);
    }

    for (uint32_t k=0; k<K; ++k) {
        const float* mu  = &means[k*dim];
        const float* var = &variances[k*dim];

        double quad = 0.0;
        for (uint32_t d=0; d<dim; ++d) {
            double df = (double)x[d] - (double)mu[d];
            quad += df*df / (double)var[d];
        }

        double logNk = (double)log_consts[k] - 0.5 * quad;
        logp[k] = (float)(log((double)weights[k]) + logNk);
    }

    float lse = safe_log_sum_exp(logp, K);
    for (uint32_t k=0; k<K; ++k) {
        post[k] = expf(logp[k] - lse);
    }

    free(logp);
}

// ================= cluster_to_vectors loader ================

typedef struct {
    uint32_t* members;
    uint32_t  count;
    uint32_t  capacity;
} cluster_members_t;

cluster_members_t* load_cluster_to_vectors(const char* path, uint32_t K) {

    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Failed to open %s\n", path);
        return NULL;
    }

    cluster_members_t* cm = calloc(K, sizeof(cluster_members_t));
    if (!cm) {
        fclose(f);
        return NULL;
    }

    char* line = NULL;
    size_t cap = 0;

    while (getline(&line, &cap, f) != -1) {

        char* save = NULL;
        char* tok = strtok_r(line, " \t\r\n", &save);
        if (!tok) continue;

        int cluster_id = atoi(tok);
        if (cluster_id < 0 || (uint32_t)cluster_id >= K) {
            fprintf(stderr,
                "Invalid cluster id %d (expected in [0, %u))\n",
                cluster_id, K);
            continue;
        }

        cluster_members_t* c = &cm[cluster_id];

        // Parse vector IDs
        tok = strtok_r(NULL, " \t\r\n", &save);
        while (tok) {
            if (c->count == c->capacity) {
                c->capacity = (c->capacity == 0) ? 1024 : c->capacity * 2;
                c->members = realloc(c->members,
                                     c->capacity * sizeof(uint32_t));
            }
            c->members[c->count++] = (uint32_t)atoi(tok);
            tok = strtok_r(NULL, " \t\r\n", &save);
        }
    }

    free(line);
    fclose(f);
    return cm;
}

// ================= distance between q and v =================

static float distance_query_train(
    const float* q,
    const float* v,
    const gmm_header_t* gh)
{
    uint32_t dim = gh->dim;
    if (gh->metric == 0) {
        // Euclidean (use squared distance for ranking)
        double sum = 0.0;
        for (uint32_t d=0; d<dim; ++d) {
            double df = (double)q[d] - (double)v[d];
            sum += df*df;
        }
        return (float)sum;
    } else {
        // Cosine distance: assume q, v are unit vectors
        double dot = 0.0;
        for (uint32_t d=0; d<dim; ++d) dot += (double)q[d]*(double)v[d];
        return (float)(1.0 - dot);  // smaller is closer
    }
}

// ================= candidate sort ==========================

typedef struct {
    uint32_t id;
    float    dist;
} candidate_t;

static int cmp_candidate(const void* a, const void* b) {
    const candidate_t* ca = (const candidate_t*)a;
    const candidate_t* cb = (const candidate_t*)b;
    if (ca->dist < cb->dist) return -1;
    if (ca->dist > cb->dist) return 1;
    return 0;
}

// ============================ MAIN =========================

int main(int argc, char** argv) {
    if (argc != 7) {
        fprintf(stderr,
            "Usage: %s <train.5bit> <test.5bit> <model.gmm> <cluster_to_vectors.index> <t_query> <t_edge>\n",
            argv[0]);
        return 1;
    }

    const char* train_path  = argv[1];
    const char* test_path   = argv[2];
    const char* gmm_path    = argv[3];
    const char* c2v_path    = argv[4];
    float t_query = (float)atof(argv[5]);
    float t_edge  = (float)atof(argv[6]);

    printf("Train .5bit:  %s\n", train_path);
    printf("Test  .5bit:  %s\n", test_path);
    printf("GMM file:     %s\n", gmm_path);
    printf("C2V index:    %s\n", c2v_path);
    printf("t_query = %.4f, t_edge = %.4f\n", t_query, t_edge);
    clock_t t_start = clock();

    // --- Load training packed vectors ---
    header_t h_train;
    uint8_t* packed_train = read_packed(train_path, &h_train);
    if (!packed_train) {
        fprintf(stderr, "Failed to read train .5bit: %s\n", train_path);
        return 1;
    }

    // --- Load test packed vectors ---
    header_t h_test;
    uint8_t* packed_test = read_packed(test_path, &h_test);
    if (!packed_test) {
        fprintf(stderr, "Failed to read test .5bit: %s\n", test_path);
        free(packed_train);
        return 1;
    }

    if (h_train.dim != h_test.dim) {
        fprintf(stderr, "Dimension mismatch train=%u test=%u\n",
                h_train.dim, h_test.dim);
        free(packed_train);
        free(packed_test);
        return 1;
    }

    uint32_t dim      = h_train.dim;
    uint32_t n_train  = h_train.n;
    uint32_t n_test   = h_test.n;

    printf("Train: n=%u dim=%u\n", n_train, dim);
    printf("Test:  n=%u dim=%u\n", n_test, dim);

    // --- Load GMM ---
    gmm_header_t gh;
    float *weights=NULL, *means=NULL, *variances=NULL, *log_consts=NULL;
    if (load_gmm(gmm_path, &gh, &weights, &means, &variances, &log_consts) != 0) {
        free(packed_train); free(packed_test);
        return 1;
    }

    if (gh.dim != dim) {
        fprintf(stderr, "GMM dim=%u but data dim=%u\n", gh.dim, dim);
        free(packed_train); free(packed_test);
        free(weights); free(means); free(variances); free(log_consts);
        return 1;
    }
    uint32_t K = gh.K;
    printf("GMM: K=%u dim=%u metric=%s normalised=%u\n",
           gh.K, gh.dim,
           (gh.metric==0 ? "euclidean" : "cosine"),
           gh.normalised);

    // --- Load cluster_to_vectors ---
    cluster_members_t* c2v = load_cluster_to_vectors(c2v_path, K);
    if (!c2v) {
        free(packed_train); free(packed_test);
        free(weights); free(means); free(variances); free(log_consts);
        return 1;
    }

    // --- Allocate scratch buffers ---
    float* buf_q  = (float*)malloc(sizeof(float)*dim);
    float* buf_v  = (float*)malloc(sizeof(float)*dim);
    float* post   = (float*)malloc(sizeof(float)*K);
    uint8_t* visited = (uint8_t*)malloc(sizeof(uint8_t)*n_train);
    uint32_t* pool   = (uint32_t*)malloc(sizeof(uint32_t)*n_train);
    candidate_t* cand = (candidate_t*)malloc(sizeof(candidate_t)*n_train);

    if (!buf_q || !buf_v || !post || !visited || !pool || !cand) {
        fprintf(stderr, "OOM for scratch buffers\n");
        free(packed_train); free(packed_test);
        free(weights); free(means); free(variances); free(log_consts);
        for (uint32_t i=0; i<K; ++i) free(c2v[i].members);
        free(c2v);
        free(buf_q); free(buf_v); free(post); free(visited); free(pool); free(cand);
        return 1;
    }

    // --- Prepare output file path ---
    ensure_dir("querying_results");
    char base[512];
    strip_ext(base_name(test_path), base, sizeof(base));

    char out_path[1024];
    snprintf(out_path, sizeof(out_path),
             "querying_results/%s_gmm_candidates.ivecs", base);

    FILE* fout = fopen(out_path, "wb");
    if (!fout) {
        fprintf(stderr, "Failed to open output ivecs: %s\n", out_path);
        // free and exit
        free(packed_train); free(packed_test);
        free(weights); free(means); free(variances); free(log_consts);
        for (uint32_t i=0; i<K; ++i) free(c2v[i].members);
        free(c2v);
        free(buf_q); free(buf_v); free(post); free(visited); free(pool); free(cand);
        return 1;
    }

    printf("Writing candidates to %s\n", out_path);

    // ==================== MAIN QUERY LOOP ====================

    for (uint32_t qi = 0; qi < n_test; ++qi) {
        // ---- 1. Dequantise query vector ----
        unpack_dequantise_vector(packed_test, &h_test, qi, buf_q);
        if (gh.metric == 1 && gh.normalised) {
            l2_normalise(buf_q, dim);
        }

        // ---- 2. Posterior P(Ck | q) ----
        posterior_probabilities(buf_q, &gh, weights, means, variances, log_consts, post);

        // ---- 3. Sort clusters by posterior, pick those >= t_query ----
        // simple O(K^2) selection since K is small (e.g., 32/64/128)
        // create index array [0..K-1]
        uint32_t* idx = (uint32_t*)malloc(sizeof(uint32_t)*K);
        for (uint32_t k=0; k<K; ++k) idx[k] = k;

        // sort idx by post[idx[k]] descending (simple insertion sort for small K)
        for (uint32_t i=1; i<K; ++i) {
            uint32_t key = idx[i];
            float keyval = post[key];
            int j = (int)i - 1;
            while (j >= 0 && post[idx[j]] < keyval) {
                idx[j+1] = idx[j];
                j--;
            }
            idx[j+1] = key;
        }

        // pick clusters with P >= t_query
        uint32_t picked_count = 0;
        for (uint32_t i=0; i<K; ++i) {
            uint32_t k = idx[i];
            if (post[k] >= t_query) {
                picked_count++;
            } else {
                break; // below threshold because sorted
            }
        }

        if (picked_count == 0) {
            // fallback: at least the best cluster
            picked_count = 1;
        }

        // ---- 4. Initial pool = union of members of picked clusters ----
        memset(visited, 0, n_train);
        uint32_t pool_size = 0;

        for (uint32_t i=0; i<picked_count; ++i) {
            uint32_t k = idx[i];
            cluster_members_t* c = &c2v[k];
            for (uint32_t j=0; j<c->count; ++j) {
                uint32_t vid = c->members[j];
                if (vid >= n_train) continue; // safety
                if (!visited[vid]) {
                    visited[vid] = 1;
                    pool[pool_size++] = vid;
                }
            }
        }

        free(idx);

        // ---- 5. Expand pool (BFS-style) using t_edge ----
        uint32_t pos = 0;
        while (pos < pool_size) {
            uint32_t vid = pool[pos++];
            // dequantise training vector
            unpack_dequantise_vector(packed_train, &h_train, vid, buf_v);
            if (gh.metric == 1 && gh.normalised) {
                l2_normalise(buf_v, dim);
            }

            // posterior over clusters for p
            posterior_probabilities(buf_v, &gh, weights, means, variances, log_consts, post);

            // for any cluster with post >= t_edge, add its members
            for (uint32_t k=0; k<K; ++k) {
                if (post[k] < t_edge) continue;
                cluster_members_t* c = &c2v[k];
                for (uint32_t j=0; j<c->count; ++j) {
                    uint32_t nb = c->members[j];
                    if (nb >= n_train) continue;
                    if (!visited[nb]) {
                        visited[nb] = 1;
                        pool[pool_size++] = nb;
                    }
                }
            }
        }

        // ---- 6. Score all pool vectors by distance to q ----
        for (uint32_t i=0; i<pool_size; ++i) {
            uint32_t vid = pool[i];
            unpack_dequantise_vector(packed_train, &h_train, vid, buf_v);
            if (gh.metric == 1 && gh.normalised) {
                l2_normalise(buf_v, dim);
            }
            cand[i].id   = vid;
            cand[i].dist = distance_query_train(buf_q, buf_v, &gh);
        }

        // ---- 7. Sort candidates by distance ----
        qsort(cand, pool_size, sizeof(candidate_t), cmp_candidate);

        // ---- 8. Take top 100 neighbours ----
        int topK = (pool_size >= 100) ? 100 : (int)pool_size;
        int32_t out_dim = 100;  // always write 100 as requested

        // write ivecs row: [int32 dim][int32 ids...]
        fwrite(&out_dim, sizeof(int32_t), 1, fout);

        // if fewer than 100, pad with the last id
        int32_t last_id = (topK > 0) ? (int32_t)cand[topK-1].id : 0;
        for (int i=0; i<100; ++i) {
            int32_t id = (i < topK) ? (int32_t)cand[i].id : last_id;
            fwrite(&id, sizeof(int32_t), 1, fout);
        }

        if ((qi+1) % 100 == 0 || qi+1 == n_test) {
            printf("Processed %u / %u queries (pool_size last=%u)\n",
                   qi+1, n_test, pool_size);
        }
    }

    fclose(fout);
    printf("Done. Wrote candidates to %s\n", out_path);

    // ---- Cleanup ----
    free(packed_train);
    free(packed_test);
    free(weights); free(means); free(variances); free(log_consts);
    for (uint32_t i=0; i<K; ++i) free(c2v[i].members);
    free(c2v);
    free(buf_q); free(buf_v); free(post); free(visited); free(pool); free(cand);

    clock_t t_end = clock();
    double ms = 1000.0 * (double)(t_end - t_start) / CLOCKS_PER_SEC;
    printf("Total query time: %.3f ms\n", ms);


    return 0;
}
