// gmm_cluster.c (fixed EM & stabilisation)
// -------------------------------------------------------------
// Soft-EM Gaussian Mixture Model over 5-bit quantised vectors (diagonal covariance).
//
// - Accuracy-first for <= 100 dims: uses DIAGONAL covariance (vs isotropic).
// - Euclidean mode: raw dequantised space.
// - Cosine mode:     L2-normalised vectors; EM runs in that space.
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
// -------------------------------------------------------------

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

static const char* SUMMARY_CSV_PATH = "analysis_results/training_summary_attempt2.csv";
static int K = 0;

// EM controls (tuned for stability)
#define EM_MAX_ITERS          200      // allow more iterations; we stop early via tolerance
#define EM_LIKELIHOOD_TOL     1e-4f    // relative improvement tolerance

// Variance regularisation
// Fraction of global variance used as a base for the floor (Euclidean mode)
#define GLOBAL_VAR_FRACTION_FLOOR  1e-4f
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
    { "quantised_data/train/glove-25-angular_train_reduced.5bit",           DIST_COSINE     },
    { "quantised_data/train/glove-50-angular_train_reduced.5bit",           DIST_COSINE     },
    { "quantised_data/train/glove-100-angular_train_reduced.5bit",          DIST_COSINE     },
    { "quantised_data/train/glove-200-angular_train_reduced.5bit",          DIST_COSINE     },
    { "quantised_data/train/lastfm-64-dot_train_reduced.5bit",              DIST_COSINE     },
    { "quantised_data/train/nytimes-256-angular_train_reduced.5bit",        DIST_COSINE     },
    { "quantised_data/train/mnist-784-euclidean_train_reduced.5bit",        DIST_EUCLIDEAN     },
    { "quantised_data/train/gist-960-euclidean_train_reduced.5bit",         DIST_EUCLIDEAN     },
    { "quantised_data/train/fashion-mnist-784-euclidean_train_reduced.5bit",    DIST_EUCLIDEAN     },
    { "quantised_data/train/sift-128-euclidean_train_reduced.5bit",          DIST_EUCLIDEAN     }
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

// We keep the farthest-first refinement but choose the very first centre randomly.
static float dist_idx_center_normpolicy(const uint8_t* packed, const header_t* h,
                                        uint32_t idx, const float* center,
                                        distance_type_t metric) {
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

    // --- c0: choose a random vector as the first centre ---
    uint32_t first = (uint32_t)(rand() % n);
    float* v = (float*)malloc(sizeof(float)*dim);
    unpack_dequantise_vector(packed, h, first, v);
    if (metric == DIST_COSINE) l2_normalise(v, dim);
    memcpy(&centers[0], v, sizeof(float)*dim);

    // distances to nearest center
    double* d2 = (double*)malloc(sizeof(double)*n);
    for (uint32_t i=0; i<n; ++i)
        d2[i] = dist_idx_center_normpolicy(packed, h, i, &centers[0], metric);

    // --- farthest-point k-means++ refinement for the remaining centres ---
    for (int k=1; k<K; ++k) {
        uint32_t far=0; double best=-1.0;
        for (uint32_t i=0; i<n; ++i) if (d2[i] > best) { best=d2[i]; far=i; }

        unpack_dequantise_vector(packed, h, far, v);
        if (metric == DIST_COSINE) l2_normalise(v, dim);
        memcpy(&centers[k*dim], v, sizeof(float)*dim);

        // update distances
        for (uint32_t i=0; i<n; ++i) {
            float dd = dist_idx_center_normpolicy(packed, h, i, &centers[k*dim], metric);
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

//=================== EM core (fixed) ===================

static int em_gmm(const uint8_t* packed, const header_t* h,
                  distance_type_t metric,
                  float* weights,
                  float* means,
                  float* variances,
                  int*   hard_assign,
                  float* out_loglike,
                  int*   out_iters)
{
    const uint32_t n   = h->n;
    const uint32_t dim = h->dim;
    const int normalised = (metric == DIST_COSINE) ? 1 : 0;

    // ---------- Initialisation ----------
    // Means via kmeans++-style
    init_kmeanspp_centers(packed, h, means, metric);

    // Initial weights uniform
    for (int k=0;k<K;++k) weights[k] = 1.0f / (float)K;

    // Compute global mean/variance to set an initial diagonal covariance and floor
    float* gmean = (float*)calloc(dim, sizeof(float));
    float* gvar  = (float*)calloc(dim, sizeof(float));
    float* tmp   = (float*)malloc(sizeof(float)*dim);
    if (!gmean || !gvar || !tmp) {
        free(gmean); free(gvar); free(tmp);
        return -1;
    }

    // global mean
    for (uint32_t i=0;i<n;++i) {
        unpack_dequantise_vector(packed, h, i, tmp);
        if (normalised) l2_normalise(tmp, dim);
        for (uint32_t d=0; d<dim; ++d) gmean[d] += tmp[d];
    }
    for (uint32_t d=0; d<dim; ++d) gmean[d] /= (float)n;

    // global variance per dim
    for (uint32_t i=0;i<n;++i) {
        unpack_dequantise_vector(packed, h, i, tmp);
        if (normalised) l2_normalise(tmp, dim);
        for (uint32_t d=0; d<dim; ++d) {
            float df = tmp[d] - gmean[d];
            gvar[d] += df*df;
        }
    }
    for (uint32_t d=0; d<dim; ++d) gvar[d] /= (float)n;

    // scalar global variance
    double gv_sum = 0.0;
    for (uint32_t d=0; d<dim; ++d) gv_sum += (double)gvar[d];
    float global_var_scalar = (float)(gv_sum / (double)dim);
    if (global_var_scalar <= 0.0f) global_var_scalar = 1e-3f;

    // variance floor: stronger for cosine, smaller for Euclidean
    float var_floor;
    if (metric == DIST_COSINE) {
        // unit-norm vectors; clusters are tight -> use a robust absolute floor
        var_floor = fmaxf(1e-3f, 0.01f * global_var_scalar);
    } else {
        var_floor = fmaxf(1e-6f,
                          GLOBAL_VAR_FRACTION_FLOOR *
                          fmaxf(global_var_scalar, 1e-6f));
    }

    // init variances to global var (per dim), with floor
    for (int k=0;k<K;++k) {
        for (uint32_t d=0; d<dim; ++d) {
            float v = gvar[d];
            if (v < var_floor) v = var_floor;
            variances[k*dim + d] = v;
        }
    }

    free(gmean);
    free(gvar);

    // responsibilities (per sample) & EM accumulators
    float* r         = (float*)malloc(sizeof(float)*K);
    float* logp      = (float*)malloc(sizeof(float)*K); // log π + log N
    float* Nk        = (float*)malloc(sizeof(float)*K);
    float* sum_mu    = (float*)malloc(sizeof(float)*K*dim);  // ∑ r_ik x_i
    float* sum_var   = (float*)malloc(sizeof(float)*K*dim);  // ∑ r_ik x_i^2
    float* inv_var   = (float*)malloc(sizeof(float)*K*dim);  // 1/σ^2
    float* log_consts= (float*)malloc(sizeof(float)*K);      // -0.5 * ∑ log(2πσ^2)

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
        memset(Nk,      0, sizeof(float)*K);
        memset(sum_mu,  0, sizeof(float)*K*dim);
        memset(sum_var, 0, sizeof(float)*K*dim);
        double total_loglike = 0.0;

        for (uint32_t i=0; i<n; ++i) {
            unpack_dequantise_vector(packed, h, i, tmp);
            if (normalised) l2_normalise(tmp, dim);

            // compute log p_k(x_i) = log π_k + log N_k(x_i)
            for (int k=0; k<K; ++k) {
                double quad = 0.0;
                const float* mu   = &means[k*dim];
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
            total_loglike += (double)logsum;

            // responsibilities and accumulators
            for (int k=0; k<K; ++k) {
                float rik = expf(logp[k] - logsum);
                r[k] = rik;
                Nk[k] += rik;
                float* sum_mu_k  = &sum_mu[k*dim];
                float* sum_var_k = &sum_var[k*dim];
                for (uint32_t d=0; d<dim; ++d) {
                    float x = tmp[d];
                    sum_mu_k[d]  += rik * x;
                    sum_var_k[d] += rik * x * x;  // we will convert to variance in M-step
                }
            }
        }

        double avg_ll = total_loglike / (double)n;
        if (out_loglike) *out_loglike = (float)avg_ll;

        // ---------- M-step ----------

        // update weights π_k
        for (int k=0;k<K;++k) {
            float wk = (Nk[k] > 0.0f) ? (Nk[k] / (float)n) : MIN_CLUSTER_WEIGHT;
            if (!isfinite(wk) || wk < MIN_CLUSTER_WEIGHT) wk = MIN_CLUSTER_WEIGHT;
            weights[k] = wk;
        }
        // renormalise to sum to 1
        float wsum = 0.0f;
        for (int k=0;k<K;++k) wsum += weights[k];
        if (wsum <= 0.0f) wsum = 1.0f;
        for (int k=0;k<K;++k) weights[k] /= wsum;

        // update means μ_k and variances σ^2_k using ∑ r x and ∑ r x^2
        for (int k=0;k<K;++k) {
            float Nk_k = fmaxf(Nk[k], 1e-6f);
            float invNk = 1.0f / Nk_k;
            float* mu_k   = &means[k*dim];
            float* s1_k   = &sum_mu[k*dim];
            float* s2_k   = &sum_var[k*dim];
            float* var_k  = &variances[k*dim];

            for (uint32_t d=0; d<dim; ++d) {
                float m  = s1_k[d] * invNk;
                float Ex2 = s2_k[d] * invNk;
                float v  = Ex2 - m*m;  // Var[x] = E[x^2] - (E[x])^2
                if (!isfinite(v) || v < var_floor) v = var_floor;

                mu_k[d]  = m;
                var_k[d] = v;
            }
        }

        // recompute inv_var and log_consts for next E-step
        for (int k=0;k<K;++k) {
            double sum_log = 0.0;
            for (uint32_t d=0; d<dim; ++d) {
                float v = fmaxf(variances[k*dim + d], var_floor);
                inv_var[k*dim + d] = 1.0f / v;
                sum_log += log(2.0*M_PI*(double)v);
            }
            log_consts[k] = (float)(-0.5 * sum_log);
        }

        // ---------- Convergence check ----------
        double rel_impr = (isfinite(prev_ll))
                        ? ((avg_ll - prev_ll) / (fabs(prev_ll) + 1e-12))
                        : INFINITY;

        printf("  [EM] iter=%d, avg_ll=%.6f, rel_impr=%.6g\n",
               it+1, (float)avg_ll, (float)rel_impr);

        if (rel_impr >= 0.0 && rel_impr < EM_LIKELIHOOD_TOL) {
            printf("  [EM] Converged after %d iterations\n", it+1);
            break;
        }
        prev_ll = avg_ll;
    }

    // Final avg log-likelihood output already set

    // ---------- Final hard assignments (using final parameters) ----------
    for (uint32_t i=0; i<n; ++i) {
        unpack_dequantise_vector(packed, h, i, tmp);
        if (normalised) l2_normalise(tmp, dim);

        for (int k=0; k<K; ++k) {
            double quad = 0.0;
            const float* mu   = &means[k*dim];
            const float* invv = &inv_var[k*dim];
            for (uint32_t d=0; d<dim; ++d) {
                double df = (double)tmp[d] - (double)mu[d];
                quad += df*df * (double)invv[d];
            }
            double logNk = (double)log_consts[k] - 0.5 * quad;
            logp[k] = (float)(log((double)fmaxf(weights[k], MIN_CLUSTER_WEIGHT)) + logNk);
        }
        int best=0; float bestv=logp[0];
        for (int k=1; k<K; ++k) if (logp[k]>bestv){bestv=logp[k];best=k;}
        hard_assign[i] = best;
    }

    free(r); free(logp); free(Nk); free(sum_mu); free(sum_var); free(inv_var); free(log_consts); free(tmp);
    if (out_iters) *out_iters = it + 1;

    return 0;
}

//==================== Per-dataset pipeline ====================

static int process_dataset(const char* filepath, distance_type_t metric) {
    int iters = 0;
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
    if (em_gmm(packed, &h, metric, weights, means, variances,
           argmax, &avg_ll, &iters) != 0) {
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




    // Ensure K-specific directories exist
    char v2c_k_dir[1024];
    char c2v_k_dir[1024];

    snprintf(v2c_k_dir, sizeof(v2c_k_dir), "gmm_indexes/vector_to_cluster/K%d", K);
    snprintf(c2v_k_dir, sizeof(c2v_k_dir), "gmm_indexes/cluster_to_vectors/K%d", K);

    ensure_dir("gmm_indexes/vector_to_cluster");
    ensure_dir("gmm_indexes/cluster_to_vectors");

    ensure_dir(v2c_k_dir);
    ensure_dir(c2v_k_dir);

    // Write vector_to_cluster → gmm_indexes/vector_to_cluster/K[K]/<base>.index
    char p_v2c[1024];
    snprintf(p_v2c, sizeof(p_v2c),
            "gmm_indexes/vector_to_cluster/K%d/%s.index", K, base);

    if (write_vector_to_cluster(p_v2c, argmax, h.n) != 0)
        fprintf(stderr, "Failed writing %s\n", p_v2c);

    // Write cluster_to_vectors → gmm_indexes/cluster_to_vectors/K[K]/<base>.index
    char p_c2v[1024];
    snprintf(p_c2v, sizeof(p_c2v),
            "gmm_indexes/cluster_to_vectors/K%d/%s.index", K, base);

    if (write_cluster_to_vectors(p_c2v, argmax, h.n) != 0)
        fprintf(stderr, "Failed writing %s\n", p_c2v);






    // Precompute log-constants with final variances for saving
    float* log_consts = (float*)malloc(sizeof(float)*K);
    if (!log_consts) {
        fprintf(stderr, "OOM allocating log_consts for saving.\n");
        free(packed); free(weights); free(means); free(variances); free(argmax);
        return -1;
    }

    // Consistent with EM: use current variances and a small absolute floor
    float var_floor_save = 1e-6f;
    for (int k=0;k<K;++k) {
        double sum_log = 0.0;
        for (uint32_t d=0; d<h.dim; ++d) {
            float v = fmaxf(variances[k*h.dim + d], var_floor_save);
            sum_log += log(2.0*M_PI * (double)v);
        }
        log_consts[k] = (float)(-0.5 * sum_log);
    }




    // Ensure directory gmm_indexes/gmm/K[K] exists
    char gmm_k_dir[1024];
    snprintf(gmm_k_dir, sizeof(gmm_k_dir), "gmm_indexes/gmm/K%d", K);

    ensure_dir("gmm_indexes/gmm");
    ensure_dir(gmm_k_dir);

    // Write .gmm to gmm_indexes/gmm/K[K]/<basename>.gmm
    char p_gmm[1024];
    snprintf(p_gmm, sizeof(p_gmm),
            "gmm_indexes/gmm/K%d/%s.gmm", K, base);

    if (write_gmm_file(p_gmm, &h, metric, (metric==DIST_COSINE)?1:0, var_floor_save,
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
            base, h.n, h.dim, K, iters, avg_ll, elapsed_sec);
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

int main(int argc, char** argv) {
    srand((unsigned)time(NULL)); // seed RNG for kmeans++ init

    // --- Parse K from command-line ---
    if (argc != 2) {
        printf("Usage: %s <K>\n", argv[0]);
        printf("Example: %s 128\n", argv[0]);
        return 1;
    }

    K = atoi(argv[1]);
    if (K <= 0) {
        fprintf(stderr, "Invalid K: %s\n", argv[1]);
        return 1;
    }

    printf("Running GMM clustering with K = %d\n", K);

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


