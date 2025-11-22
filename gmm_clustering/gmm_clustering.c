// gmm_cluster.c
// Soft-EM Gaussian Mixture Model over 5-bit quantised vectors (diagonal covariance).
//
// - Accuracy-first for <= 100 dims: uses DIAGONAL covariance (vs isotropic).
// - Euclidean mode: raw dequantised space.
// - Cosine mode:     L2-normalised vectors; EM runs in that space (typical).
//
// COMPILING:
//   gcc -O2 gmm_clustering/gmm_clustering.c 5bit_quantisation/quant_functions.c -lm -o gmm_cluster
//
// RUNNING:
//   ./gmm_cluster
//
// OUTPUT (per dataset; basename = file name without path/extension):
//   gmm_indexes/vector_to_cluster/<basename>.index    // line i: "<i> <argmax_k r_ik>"
//   gmm_indexes/cluster_to_vectors/<basename>.index   // line per cluster: "k v0 v1 ..."
//   gmm_indexes/gmm/<basename>.gmm                    // binary: header + params + precomputed terms
//
// .gmm layout (for querying P(k | x)):
//   struct gmm_file_header {
//       uint32_t K;
//       uint32_t dim;
//       uint32_t metric;       // 0=EUCLIDEAN, 1=COSINE
//       uint32_t normalised;   // 0/1 (COSINE mode uses 1)
//       float    var_floor;    // variance floor used during training
//   }
//   float weights[K];                 // π_k
//   float means[K*dim];               // μ_k (row-major: k then d)
//   float variances[K*dim];           // diag Σ_k[d]
//   float log_const[K];               // -0.5 * (sum_d log(2πσ_kd^2))  (for fast per-query log-likelihood)
//
// During querying (from floats or 5-bit), do the same normalisation if metric==COSINE,
// then compute for each k:
//   logN_k(x) = log_const[k] - 0.5 * sum_d ( (x_d - μ_kd)^2 / σ_kd^2 )
//   log p_k(x) = log(π_k) + logN_k(x)
//   posterior r_k(x) = softmax_k log p_k(x)
//

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <float.h>
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

//======================= CONFIG =======================

// Global, easy-to-spot K (number of clusters)
static int K = 64;

static const char* SUMMARY_CSV_PATH = "analysis_results/training_summary.csv";

// EM controls
#define EM_MAX_ITERS          50       // hard cap
#define EM_LIKELIHOOD_TOL     1e-4f    // relative improvement tolerance
#define INIT_KMEANSPP_FAR_FIRST 1      // deterministic farthest-first kmeans++
#define EMPTY_THRESHOLD     0.5f        // consider cluster empty if Nk < 0.5
#define SPLIT_THRESHOLD     2.0f        // split if Nk > SPLIT_THRESHOLD * average cluster size
#define PERTURB_SCALE       0.05f       // small random shift when splitting

// Variance regularisation
#define GLOBAL_VAR_FRACTION_FLOOR  1e-6f   // add this * global_var to each variance dim
#define MIN_CLUSTER_WEIGHT         1e-3f   // to avoid zero weights


typedef struct {
    char dataset_name[256];
    uint32_t n;
    uint32_t dim;
    int K;
    int iterations;
    float avg_loglike;
    double time_sec;
} training_result_t;


//================== Dataset list (edit) ==================

typedef struct {
    const char* path;
    distance_type_t metric; // DIST_EUCLIDEAN or DIST_COSINE
} dataset_spec_t;

// Example list — edit to your files
static dataset_spec_t datasets[] = {
    { "quantised_data/train/coco-i2i-512-angular_train_reduced.5bit",       DIST_COSINE     },
    { "quantised_data/train/glove-25-angular_train_reduced.5bit",       DIST_COSINE     },
    { "quantised_data/train/glove-50-angular_train_reduced.5bit",       DIST_COSINE     },
    { "quantised_data/train/glove-100-angular_train_reduced.5bit",          DIST_COSINE     },
    { "quantised_data/train/glove-200-angular_train_reduced.5bit",          DIST_COSINE     },
    { "quantised_data/train/lastfm-64-dot_train_reduced.5bit",          DIST_COSINE     },
    { "quantised_data/train/nytimes-256-angular_train_reduced.5bit",          DIST_COSINE     }
};

//================ Path helpers =================

static void ensure_dir(const char* path) {
    if (MKDIR(path) != 0) { (void)errno; } // ok if exists
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
    if (i == 0) snprintf(out, outsz, "%s", name);
    else {
        size_t base_len = i-1;
        if (base_len >= outsz) base_len = outsz-1;
        memcpy(out, name, base_len);
        out[base_len] = '\0';
    }
}

//================ Dequantise / normalise helpers =================

static void unpack_dequantise_vector(const uint8_t* packed, const header_t* h,
                                     uint32_t idx, float* out) {
    const uint32_t dim = h->dim;
    const float minv = h->min_val;
    const float range = (h->max_val - h->min_val);
    const uint32_t start_bit = idx * dim * BIT_DEPTH;

    if (range == 0.0f) { for (uint32_t d=0; d<dim; ++d) out[d]=minv; return; }

    for (uint32_t d = 0; d < dim; ++d) {
        uint32_t bit_pos = start_bit + d * BIT_DEPTH;
        uint8_t q = 0;
        for (uint8_t b = 0; b < BIT_DEPTH; ++b) {
            uint32_t byte_idx = (bit_pos + b) / 8;
            uint8_t bit_off   = (bit_pos + b) % 8;
            if (packed[byte_idx] & (1u << bit_off)) q |= (1u << b);
        }
        out[d] = minv + (range * q) / (LEVELS - 1);
    }
}

static void l2_normalise(float* v, uint32_t dim) {
    double nrm = 0.0;
    for (uint32_t d=0; d<dim; ++d) nrm += (double)v[d]*(double)v[d];
    nrm = sqrt(nrm);
    if (nrm > 0.0) {
        float inv = (float)(1.0/nrm);
        for (uint32_t d=0; d<dim; ++d) v[d]*=inv;
    }
}

//================ Distance functions ================

float euclidean_distance(const uint8_t* packed, uint32_t idx1, uint32_t idx2, const header_t* h) {
    const uint32_t dim = h->dim;
    float* v1 = (float*)malloc(sizeof(float)*dim);
    float* v2 = (float*)malloc(sizeof(float)*dim);
    if (!v1 || !v2) { free(v1); free(v2); return 0.0f; }
    unpack_dequantise_vector(packed, h, idx1, v1);
    unpack_dequantise_vector(packed, h, idx2, v2);
    double sum=0.0; for (uint32_t d=0; d<dim; ++d){ double df=(double)v1[d]-(double)v2[d]; sum+=df*df; }
    free(v1); free(v2);
    return (float)sqrt(sum);
}

float cosine_distance(const uint8_t* packed, uint32_t idx1, uint32_t idx2, const header_t* h) {
    const uint32_t dim = h->dim;
    float* v1 = (float*)malloc(sizeof(float)*dim);
    float* v2 = (float*)malloc(sizeof(float)*dim);
    if (!v1 || !v2) { free(v1); free(v2); return 0.0f; }
    unpack_dequantise_vector(packed, h, idx1, v1);
    unpack_dequantise_vector(packed, h, idx2, v2);
    // normalise
    l2_normalise(v1, dim);
    l2_normalise(v2, dim);
    double dot=0.0; for (uint32_t d=0; d<dim; ++d) dot+=(double)v1[d]*(double)v2[d];
    free(v1); free(v2);
    double sim = dot; // unit vectors
    return (float)(1.0 - sim);
}

//================ K-means++-style init (centres only) ================

static float dist_idx_center_normpolicy(const uint8_t* packed, const header_t* h,
                                        uint32_t idx, const float* center,
                                        distance_type_t metric, int center_is_unit) {
    const uint32_t dim = h->dim;
    float* v = (float*)malloc(sizeof(float)*dim);
    if (!v) return 0.0f;
    unpack_dequantise_vector(packed, h, idx, v);
    if (metric == DIST_COSINE) l2_normalise(v, dim);

    double acc=0.0;
    if (metric == DIST_EUCLIDEAN) {
        for (uint32_t d=0; d<dim; ++d){ double df=(double)v[d]-(double)center[d]; acc+=df*df; }
        free(v);
        return (float)sqrt(acc);
    } else {
        double dot=0.0, nv=0.0, nc=0.0;
        for (uint32_t d=0; d<dim; ++d) {
            dot += (double)v[d]*(double)center[d];
            nv  += (double)v[d]*(double)v[d];
            nc  += (double)center[d]*(double)center[d];
        }
        free(v);
        double denom = sqrt(nv)*sqrt(nc);
        double sim = denom>0.0? dot/denom : 1.0;
        return (float)(1.0 - sim);
    }
}

static void init_kmeanspp_centers(const uint8_t* packed, const header_t* h,
                                  float* centers, distance_type_t metric) {
    const uint32_t n = h->n, dim = h->dim;
    if (n==0) return;

    // c0
    uint32_t first = 0;
    float* v = (float*)malloc(sizeof(float)*dim);
    unpack_dequantise_vector(packed, h, first, v);
    if (metric == DIST_COSINE) l2_normalise(v, dim);
    memcpy(&centers[0], v, sizeof(float)*dim);

    // distances to nearest center
    double* d2 = (double*)malloc(sizeof(double)*n);
    for (uint32_t i=0; i<n; ++i)
        d2[i] = dist_idx_center_normpolicy(packed, h, i, &centers[0], metric, 1);

    for (int k=1; k<K; ++k) {
        // deterministic farthest-point choice
        uint32_t far=0; double best=-1.0;
        for (uint32_t i=0; i<n; ++i) if (d2[i] > best) { best=d2[i]; far=i; }

        unpack_dequantise_vector(packed, h, far, v);
        if (metric == DIST_COSINE) l2_normalise(v, dim);
        memcpy(&centers[k*dim], v, sizeof(float)*dim);

        // update distances
        for (uint32_t i=0; i<n; ++i) {
            float dd = dist_idx_center_normpolicy(packed, h, i, &centers[k*dim], metric, 1);
            if (dd < d2[i]) d2[i] = dd;
        }
    }

    free(v); free(d2);
}

//================ EM utilities =================

static float safe_log_sum_exp(const float* arr, int Kc) {
    // compute log(sum exp(arr)))
    float maxv = arr[0];
    for (int k=1;k<Kc;++k) if (arr[k] > maxv) maxv = arr[k];
    double sum = 0.0;
    for (int k=0;k<Kc;++k) sum += exp((double)arr[k] - (double)maxv);
    return (float)(log(sum) + (double)maxv);
}

static void add_vec(float* dst, const float* src, uint32_t dim, float scale) {
    for (uint32_t d=0; d<dim; ++d) dst[d] += scale * src[d];
}

static void zero_vec(float* v, uint32_t dim) { memset(v, 0, sizeof(float)*dim); }

//================ GMM file writer =================

typedef struct {
    uint32_t K;
    uint32_t dim;
    uint32_t metric;     // 0=EUCLIDEAN, 1=COSINE
    uint32_t normalised; // 0/1
    float    var_floor;  // variance floor used
} gmm_file_header;

static int write_gmm_file(const char* filepath,
                          const header_t* h,
                          distance_type_t metric,
                          int normalised,
                          float var_floor,
                          const float* weights,     // K
                          const float* means,       // K*dim
                          const float* variances,   // K*dim
                          float* log_consts) {      // K
    FILE* f = fopen(filepath, "wb");
    if (!f) { fprintf(stderr, "Failed to open %s for writing\n", filepath); return -1; }

    gmm_file_header hd;
    hd.K = (uint32_t)K;
    hd.dim = h->dim;
    hd.metric = (metric==DIST_EUCLIDEAN?0:1);
    hd.normalised = (uint32_t)normalised;
    hd.var_floor = var_floor;

    fwrite(&hd, sizeof(hd), 1, f);
    fwrite(weights,   sizeof(float), (size_t)K,         f);
    fwrite(means,     sizeof(float), (size_t)K*h->dim,  f);
    fwrite(variances, sizeof(float), (size_t)K*h->dim,  f);
    fwrite(log_consts,sizeof(float), (size_t)K,         f);

    fclose(f);
    return 0;
}

//================ Index writers =================

static int write_vector_to_cluster(const char* path, const int* argmax_assign, uint32_t n) {
    FILE* f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Failed to open %s\n", path); return -1; }
    for (uint32_t i=0;i<n;++i) fprintf(f, "%u %d\n", i, argmax_assign[i]);
    fclose(f);
    return 0;
}

static int write_cluster_to_vectors(const char* path, const int* argmax_assign, uint32_t n) {
    FILE* f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Failed to open %s\n", path); return -1; }
    for (int k=0;k<K;++k) {
        fprintf(f, "%d", k);
        for (uint32_t i=0;i<n;++i) if (argmax_assign[i]==k) fprintf(f, " %u", i);
        fprintf(f, "\n");
    }
    fclose(f);
    return 0;
}

// ================== Cluster Balancing ===================

static int find_smallest_cluster(const float* Nk, int K) {
    int min_k = 0;
    for (int k = 1; k < K; ++k)
        if (Nk[k] < Nk[min_k]) min_k = k;
    return min_k;
}

static uint32_t farthest_point_index(const uint8_t* packed, const header_t* h,
                                     const float* means, int K, distance_type_t metric) {
    // Returns index of the sample farthest from all cluster means
    uint32_t n = h->n, dim = h->dim;
    float* tmp = (float*)malloc(sizeof(float) * dim);
    if (!tmp) return rand() % n;

    double best_dist = -1.0;
    uint32_t best_idx = 0;
    for (uint32_t i = 0; i < n; i += (n / 500 + 1)) {  // sample subset for efficiency
        unpack_dequantise_vector(packed, h, i, tmp);
        if (metric == DIST_COSINE) l2_normalise(tmp, dim);

        double min_dist = DBL_MAX;
        for (int k = 0; k < K; ++k) {
            double dist = 0.0;
            const float* mu = &means[k * dim];
            for (uint32_t d = 0; d < dim; ++d) {
                double df = (double)tmp[d] - (double)mu[d];
                dist += df * df;
            }
            if (dist < min_dist) min_dist = dist;
        }

        if (min_dist > best_dist) {
            best_dist = min_dist;
            best_idx = i;
        }
    }
    free(tmp);
    return best_idx;
}

static void maintain_cluster_balance(float* weights, float* means, float* variances,
                                     float* Nk, uint32_t dim, int K, int n,
                                     const uint8_t* packed, const header_t* h,
                                     distance_type_t metric, float* var_floor)
{
    // ---- Adaptive variance floor ----
    *var_floor *= 1.2f;        // increase variance 20% per EM cycle
    if (*var_floor > 1e-1f)    // allow up to 0.1
        *var_floor = 1e-1f;

    float avg_size = (float)n / (float)K;
    int performed_split = 0;

    // ---- 1. Reinitialise empties using farthest-point ----
    for (int k = 0; k < K; ++k) {
        if (Nk[k] < EMPTY_THRESHOLD) {
            uint32_t idx = farthest_point_index(packed, h, means, K, metric);
            float* v = (float*)malloc(sizeof(float) * dim);
            unpack_dequantise_vector(packed, h, idx, v);
            if (metric == DIST_COSINE) l2_normalise(v, dim);
            memcpy(&means[k * dim], v, sizeof(float) * dim);
            free(v);

            for (uint32_t d = 0; d < dim; ++d)
                variances[k * dim + d] = fmaxf(1e-3f, *var_floor);
            weights[k] = 1.0f / (float)K;
            Nk[k] = avg_size;
            printf("[Reinit] Empty cluster %d reinitialised (farthest-point)\n", k);
        }
    }

    // ---- 2. Split all oversized clusters ----
    for (int k = 0; k < K; ++k) {
        if (Nk[k] > SPLIT_THRESHOLD * avg_size) {
            int new_k = find_smallest_cluster(Nk, K);
            if (new_k == k) continue;

            memcpy(&means[new_k * dim], &means[k * dim], sizeof(float) * dim);
            memcpy(&variances[new_k * dim], &variances[k * dim], sizeof(float) * dim);

            // Perturb in opposite directions
            for (uint32_t d = 0; d < dim; ++d) {
                float delta = PERTURB_SCALE * ((float)rand() / RAND_MAX - 0.5f) *
                              sqrtf(fmaxf(variances[k * dim + d], *var_floor));
                means[k * dim + d] += delta;
                means[new_k * dim + d] -= delta;
            }

            // Soft redistribution
            float parent_w = weights[k];
            weights[k] = parent_w * 0.6f;
            weights[new_k] = parent_w * 0.4f;
            Nk[new_k] = Nk[k] * 0.4f;
            Nk[k] *= 0.6f;

            performed_split++;
            printf("[Split] Cluster %d split -> new cluster %d (Nk=%.1f)\n",
                   k, new_k, Nk[new_k]);
        }
    }

    if (performed_split == 0)
        return;
    
    if (metric == DIST_COSINE) {
        for (int k = 0; k < K; ++k)
            l2_normalise(&means[k * dim], dim);
    }

    // ---- 3. Normalise weights ----
    float sumw = 0.0f;
    for (int k = 0; k < K; ++k) sumw += weights[k];
    for (int k = 0; k < K; ++k) weights[k] /= fmaxf(sumw, 1e-12f);
}


// ================== Cluster Reinitialisation ===================

static void reinitialise_degenerate_clusters(const uint8_t* packed, const header_t* h,
                                             float* weights, float* means, float* variances,
                                             float* Nk, uint32_t dim, int K,
                                             distance_type_t metric)
{
    float global_var = 0.0f;
    for (int k = 0; k < K; ++k) {
        for (uint32_t d = 0; d < dim; ++d)
            global_var += variances[k * dim + d];
    }
    global_var /= (float)(K * dim);
    if (global_var <= 0.0f) global_var = 1e-6f;

    for (int k = 0; k < K; ++k) {
        int degenerate = 0;
        if (!isfinite(weights[k]) || weights[k] < 1e-7f)
            degenerate = 1;
        else {
            // Check for NaN or zero variance
            for (uint32_t d = 0; d < dim; ++d) {
                if (!isfinite(variances[k * dim + d]) || variances[k * dim + d] < 1e-12f) {
                    degenerate = 1;
                    break;
                }
            }
        }

        if (degenerate) {
            // Randomly sample a vector to re-seed
            uint32_t idx = rand() % h->n;
            float* v = (float*)malloc(sizeof(float) * dim);
            if (!v) continue;
            unpack_dequantise_vector(packed, h, idx, v);
            if (metric == DIST_COSINE) l2_normalise(v, dim);

            memcpy(&means[k * dim], v, sizeof(float) * dim);
            for (uint32_t d = 0; d < dim; ++d)
                variances[k * dim + d] = global_var;

            weights[k] = 1.0f / (float)K;
            Nk[k] = (float)(h->n / K);
            free(v);

            printf("[Reinit] Cluster %d reinitialised (degenerate detected)\n", k);
        }
    }

    // Renormalise weights
    float sumw = 0.0f;
    for (int k = 0; k < K; ++k)
        sumw += weights[k];
    for (int k = 0; k < K; ++k)
        weights[k] /= fmaxf(sumw, 1e-12f);
}

//=================== EM core ===================

static int em_gmm(const uint8_t* packed, const header_t* h,
                  distance_type_t metric,
                  float* weights,      // out: K
                  float* means,        // out: K*dim
                  float* variances,    // out: K*dim
                  int*   hard_assign,  // out: n (argmax_k responsibilities)
                  float* out_loglike)  // out: final avg log-likelihood
{
    const uint32_t n   = h->n;
    const uint32_t dim = h->dim;
    const int normalised = (metric == DIST_COSINE) ? 1 : 0;

    // ---------- Initialisation ----------
    // Means via kmeans++-style
    init_kmeanspp_centers(packed, h, means, metric);

    // Initial weights uniform
    for (int k=0;k<K;++k) weights[k] = 1.0f / (float)K;

    // Initial variances from global variance (per dim)
    // Compute global mean and var as a rough start
    float* gmean = (float*)calloc(dim, sizeof(float));
    float* tmp = (float*)malloc(sizeof(float)*dim);
    if (!gmean || !tmp) { free(gmean); free(tmp); return -1; }

    // global mean
    for (uint32_t i=0;i<n;++i) {
        unpack_dequantise_vector(packed, h, i, tmp);
        if (normalised) l2_normalise(tmp, dim);
        add_vec(gmean, tmp, dim, 1.0f);
    }
    for (uint32_t d=0; d<dim; ++d) gmean[d] /= (float)n;

    // global var
    float* gvar = (float*)calloc(dim, sizeof(float));
    for (uint32_t i=0;i<n;++i) {
        unpack_dequantise_vector(packed, h, i, tmp);
        if (normalised) l2_normalise(tmp, dim);
        for (uint32_t d=0; d<dim; ++d) {
            float df = tmp[d] - gmean[d];
            gvar[d] += df*df;
        }
    }
    for (uint32_t d=0; d<dim; ++d) gvar[d] = (gvar[d] / (float)n);

    // init variances to global var (avoid zeros)
    for (int k=0;k<K;++k) {
        memcpy(&variances[k*dim], gvar, sizeof(float)*dim);
    }

    // variance floor (absolute) based on global var
    double gv_sum = 0.0;
    for (uint32_t d=0; d<dim; ++d) gv_sum += (double)gvar[d];
    float global_var_scalar = (float)(gv_sum / (double)dim);
    // --- Stronger starting floor for cosine datasets ---
    float var_floor = (metric == DIST_COSINE)
        ? 1e-3f   // higher initial variance for unit-norm data
        : fmaxf(1e-12f,
            GLOBAL_VAR_FRACTION_FLOOR * fmaxf(global_var_scalar, 1e-12f));

    free(gmean);
    free(gvar);

    // responsibilities buffer (per sample)
    float* r = (float*)malloc(sizeof(float)*K);
    float* logp = (float*)malloc(sizeof(float)*K); // log π + log N
    float* Nk   = (float*)malloc(sizeof(float)*K);
    float* sum_mu = (float*)malloc(sizeof(float)*K*dim);     // ∑ r_ik x_i
    float* sum_var = (float*)malloc(sizeof(float)*K*dim);    // ∑ r_ik (x_i - μ_k)^2
    float* inv_var = (float*)malloc(sizeof(float)*K*dim);    // 1/σ^2
    float* log_consts = (float*)malloc(sizeof(float)*K);     // -0.5 * ∑ log(2πσ^2)

    if (!r || !logp || !Nk || !sum_mu || !sum_var
        || !inv_var || !log_consts) {
        free(r); free(logp); free(Nk); free(sum_mu);
        free(sum_var); free(inv_var); free(log_consts);
        free(tmp);
        return -1;
    }

    // Precompute inv_var & log_const for first iteration
    for (int k=0;k<K;++k) {
        double sum_log = 0.0;
        for (uint32_t d=0; d<dim; ++d) {
            float v = fmaxf(variances[k*dim+d], var_floor);
            inv_var[k*dim+d] = 1.0f / v;
            sum_log += log(2.0*M_PI*(double)v);
        }
        log_consts[k] = (float)(-0.5 * sum_log);
    }

    double prev_ll = -INFINITY;
    int it = 0;
    for (it=0; it<EM_MAX_ITERS; ++it) {

        // ---------- E-step ----------
        double total_loglike = 0.0;
        memset(Nk, 0, sizeof(float)*K);
        memset(sum_mu, 0, sizeof(float)*K*dim);
        memset(sum_var, 0, sizeof(float)*K*dim);

        for (uint32_t i=0; i<n; ++i) {
            unpack_dequantise_vector(packed, h, i, tmp);
            if (normalised) l2_normalise(tmp, dim);

            // compute log p_k(x_i) = log π_k + log N_k(x_i)
            for (int k=0; k<K; ++k) {
                // Mahalanobis with diagonal Σ
                double quad = 0.0;
                const float* mu = &means[k*dim];
                const float* invv = &inv_var[k*dim];
                for (uint32_t d=0; d<dim; ++d) {
                    double df = (double)tmp[d] - (double)mu[d];
                    quad += df*df * (double)invv[d];
                }
                double logNk = (double)log_consts[k] - 0.5 * quad;
                logp[k] = (float)(log((double)fmaxf(weights[k],
                    MIN_CLUSTER_WEIGHT)) + logNk);
            }

            float logsum = safe_log_sum_exp(logp, K);
            // responsibilities
            for (int k=0; k<K; ++k) {
                r[k] = expf(logp[k] - logsum);
            }
            total_loglike += (double)logsum;

            // accumulate Nk, sum_mu (for means)
            for (int k=0;k<K;++k) {
                float rik = r[k];
                Nk[k] += rik;
                add_vec(&sum_mu[k*dim], tmp, dim, rik);
            }
        }

        double avg_ll = total_loglike / (double)n;
        if (out_loglike) *out_loglike = (float)avg_ll;

        // ---------- M-step ----------
        // update weights
        for (int k=0;k<K;++k) {
            weights[k] = fmaxf(Nk[k] / (float)n, MIN_CLUSTER_WEIGHT);
        }
        // renormalise weights to sum 1
        float wsum = 0.0f; for (int k=0;k<K;++k) wsum += weights[k];
        for (int k=0;k<K;++k) weights[k] /= fmaxf(wsum, 1e-12f);

        // update means μ_k = sum_mu / Nk
        for (int k=0;k<K;++k) {
            float invNk = (Nk[k] > 0.0f) ? (1.0f/Nk[k]) : 0.0f;
            for (uint32_t d=0; d<dim; ++d) {
                means[k*dim+d] = sum_mu[k*dim+d] * invNk;
            }
            if (normalised) {
                // keep means normalised too (common in cosine setups)
                l2_normalise(&means[k*dim], dim);
            }
        }

        // recompute variances needs (x - μ_k)^2 accumulations -> second pass
        for (uint32_t i=0; i<n; ++i) {
            unpack_dequantise_vector(packed, h, i, tmp);
            if (normalised) l2_normalise(tmp, dim);

            // compute responsibilities again (with updated μ only affects M-var; accurate approach is to recompute r)
            // For strict EM, this should use previous r (already used for μ). Here we recompute to approximate using updated μ,
            // which is commonly acceptable and keeps one responsibility buffer instead of storing all r_ik.
            for (int k=0; k<K; ++k) {
                double quad = 0.0;
                const float* mu = &means[k*dim];
                const float* invv = &inv_var[k*dim]; // still old inv_var; fine for M-step var computation
                for (uint32_t d=0; d<dim; ++d) {
                    double df = (double)tmp[d] - (double)mu[d];
                    quad += df*df * (double)invv[d];
                }
                double logNk = (double)log_consts[k] - 0.5 * quad;
                logp[k] = (float)(log((double)fmaxf(weights[k],
                    MIN_CLUSTER_WEIGHT)) + logNk);
            }
            float logsum2 = safe_log_sum_exp(logp, K);
            for (int k=0; k<K; ++k) r[k] = expf(logp[k] - logsum2);

            for (int k=0; k<K; ++k) {
                float rik = r[k];
                const float* mu = &means[k*dim];
                for (uint32_t d=0; d<dim; ++d) {
                    float df = tmp[d] - mu[d];
                    sum_var[k*dim+d] += rik * df*df;
                }
            }
        }

        // set variances = sum_var / Nk  (with floor)
        for (int k=0;k<K;++k) {
            float invNk = (Nk[k] > 0.0f) ? (1.0f/Nk[k]) : 0.0f;
            for (uint32_t d=0; d<dim; ++d) {
                float v = sum_var[k*dim+d] * invNk;
                if (v < var_floor) v = var_floor;
                variances[k*dim+d] = v;
            }
        }

        // update inv_var and log_consts for next E-step
        for (int k=0;k<K;++k) {
            double sum_log = 0.0;
            for (uint32_t d=0; d<dim; ++d) {
                float v = fmaxf(variances[k*dim+d], var_floor);
                inv_var[k*dim+d] = 1.0f / v;
                sum_log += log(2.0*M_PI*(double)v);
            }
            log_consts[k] = (float)(-0.5 * sum_log);
        }

        if (it % 2 == 0)
            maintain_cluster_balance(weights, means, variances,
                Nk, dim, K, n, packed, h, metric, &var_floor);


        // --- Reinitialise degenerate clusters if needed ---
        reinitialise_degenerate_clusters(packed, h, weights, means,
            variances, Nk, dim, K, metric);

        // ---------- Convergence check ----------
        double rel_impr = (isfinite(prev_ll)) ? ( (avg_ll - prev_ll) / (fabs(prev_ll) + 1e-12) ) : INFINITY;
        if (rel_impr >= 0.0 && rel_impr < EM_LIKELIHOOD_TOL) break;
        prev_ll = avg_ll;
    }

    // Final hard assignments (argmax responsibilities) for index files
    for (uint32_t i=0; i<n; ++i) {
        unpack_dequantise_vector(packed, h, i, tmp);
        if (normalised) l2_normalise(tmp, dim);

        for (int k=0; k<K; ++k) {
            double quad = 0.0;
            const float* mu = &means[k*dim];
            const float* invv = &inv_var[k*dim];
            for (uint32_t d=0; d<dim; ++d) {
                double df = (double)tmp[d] - (double)mu[d];
                quad += df*df * (double)invv[d];
            }
            double logNk = (double)log_consts[k] - 0.5 * quad;
            logp[k] = (float)(log((double)fmaxf(weights[k], MIN_CLUSTER_WEIGHT)) + logNk);
        }
        // argmax
        int best=0; float bestv=logp[0];
        for (int k=1; k<K; ++k) if (logp[k]>bestv){bestv=logp[k];best=k;}
        hard_assign[i] = best;
    }

    free(r); free(logp); free(Nk); free(sum_mu); free(sum_var); free(inv_var); free(log_consts); free(tmp);
    return 0;
}

//==================== Per-dataset pipeline ====================

static int process_dataset(const char* filepath, distance_type_t metric) {
    clock_t start_time = clock();
    header_t h;
    uint8_t* packed = read_packed(filepath, &h);
    if (!packed) { fprintf(stderr, "Failed to read %s\n", filepath); return -1; }

    if (K <= 0) { fprintf(stderr, "Invalid K\n"); free(packed); return -1; }
    if ((uint32_t)K > h.n) { fprintf(stderr, "K (%d) > n (%u). Reduce K.\n", K, h.n); free(packed); return -1; }

    printf("Dataset: %s (n=%u, dim=%u, range=[%.5f, %.5f], metric=%s)\n",
           filepath, h.n, h.dim, h.min_val, h.max_val,
           (metric==DIST_EUCLIDEAN?"euclidean":"cosine"));

    // Allocate model holders
    float* weights   = (float*)malloc(sizeof(float)*K);
    float* means     = (float*)malloc(sizeof(float)*K*h.dim);
    float* variances = (float*)malloc(sizeof(float)*K*h.dim);
    int*   argmax    = (int*)  malloc(sizeof(int)*h.n);
    float  avg_ll    = 0.0f;

    if (!weights || !means || !variances || !argmax) {
        fprintf(stderr, "OOM allocating model buffers.\n");
        free(packed); free(weights); free(means); free(variances); free(argmax);
        return -1;
    }

    // Run EM
    if (em_gmm(packed, &h, metric, weights, means, variances, argmax, &avg_ll) != 0) {
        fprintf(stderr, "EM failed for %s\n", filepath);
        free(packed); free(weights); free(means); free(variances); free(argmax);
        return -1;
    }
    printf("EM complete. Avg log-likelihood: %.6f\n", avg_ll);

    clock_t end_time = clock();
    double elapsed_sec = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Training time: %.3f seconds\n", elapsed_sec);

    // Ensure output dirs
    ensure_dir("gmm_indexes");
    ensure_dir("gmm_indexes/vector_to_cluster");
    ensure_dir("gmm_indexes/cluster_to_vectors");
    ensure_dir("gmm_indexes/gmm");

    // base name
    char base[512]; strip_ext(base_name(filepath), base, sizeof(base));

    // Write vector_to_cluster
    char p_v2c[1024]; snprintf(p_v2c, sizeof(p_v2c), "gmm_indexes/vector_to_cluster/%s.index", base);
    if (write_vector_to_cluster(p_v2c, argmax, h.n) != 0)
        fprintf(stderr, "Failed writing %s\n", p_v2c);

    // Write cluster_to_vectors
    char p_c2v[1024]; snprintf(p_c2v, sizeof(p_c2v), "gmm_indexes/cluster_to_vectors/%s.index", base);
    if (write_cluster_to_vectors(p_c2v, argmax, h.n) != 0)
        fprintf(stderr, "Failed writing %s\n", p_c2v);

    // Precompute log-constants with final variances for saving
    float* log_consts = (float*)malloc(sizeof(float)*K);
    float var_floor = 0.0f; // (stored for reference) The per-iter floor used is embedded; we store 0 here
    // Recompute log_consts the same way as in EM (consistent)
    for (int k=0;k<K;++k) {
        double sum_log = 0.0;
        for (uint32_t d=0; d<h.dim; ++d)
            sum_log += log(2.0*M_PI * (double)fmaxf(variances[k*h.dim + d], 1e-12f));
        log_consts[k] = (float)(-0.5 * sum_log);
    }

    // Write .gmm
    char p_gmm[1024]; snprintf(p_gmm, sizeof(p_gmm), "gmm_indexes/gmm/%s.gmm", base);
    if (write_gmm_file(p_gmm, &h, metric, (metric==DIST_COSINE)?1:0, var_floor,
                       weights, means, variances, log_consts) != 0) {
        fprintf(stderr, "Failed writing %s\n", p_gmm);
    } else {
        printf("Wrote GMM file: %s\n", p_gmm);
    }

    // ---- Append timing + metadata to CSV ----
    FILE* fcsv = fopen(SUMMARY_CSV_PATH, "a");
    if (fcsv) {
        // If file is empty, write header first
        fseek(fcsv, 0, SEEK_END);
        if (ftell(fcsv) == 0) {
            fprintf(fcsv, "dataset,n,dim,K,iterations,avg_loglike,time_sec\n");
        }

        fprintf(fcsv, "%s,%u,%u,%d,%d,%.6f,%.3f\n",
                base, h.n, h.dim, K, EM_MAX_ITERS, avg_ll, elapsed_sec);
        fclose(fcsv);
        printf("Logged summary for %s to %s\n", base, SUMMARY_CSV_PATH);
    } else {
        fprintf(stderr, "Failed to write summary CSV: %s\n", SUMMARY_CSV_PATH);
    }


    free(log_consts);
    free(packed); free(weights); free(means); free(variances); free(argmax);
    return 0;
}

//==================== MAIN ====================

int main(void) {
    const size_t num = sizeof(datasets)/sizeof(datasets[0]);
    for (size_t i=0; i<num; ++i) {
        printf("==== Processing %s ====\n", datasets[i].path);
        if (process_dataset(datasets[i].path, datasets[i].metric) != 0) {
            fprintf(stderr, "Error processing %s\n", datasets[i].path);
        }
    }
    printf("All done.\n");
    return 0;
}
