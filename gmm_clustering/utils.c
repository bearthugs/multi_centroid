#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include "gmm_cluster_runner.h"
#include "../quant_functions.h"  // for BIT_DEPTH, LEVELS

#define TOP_CLUSTERS 3  // number of clusters to return (can be changed)

/* ===========================
 * 1️⃣ LOAD SVD MODEL & REDUCE
 * ===========================
 * The SVD model file (e.g. coco-i2i-512-angular_train_svd_model.pkl)
 * is assumed to be saved as a binary dump containing:
 *     - uint32_t orig_dim
 *     - uint32_t reduced_dim
 *     - float[orig_dim * reduced_dim] (row-major)
 *
 * The function reads that, multiplies input vector by the projection matrix,
 * and returns a reduced float vector (caller frees it).
 */

// Reduce a vector using pretrained SVD projection
float* reduce_vector_with_svd(const float* vec, uint32_t orig_dim, const char* model_path) {
    FILE* f = fopen(model_path, "rb");
    if (!f) {
        perror("Failed to open SVD model file");
        return NULL;
    }

    uint32_t model_orig_dim, reduced_dim;
    fread(&model_orig_dim, sizeof(uint32_t), 1, f);
    fread(&reduced_dim, sizeof(uint32_t), 1, f);

    if (model_orig_dim != orig_dim) {
        fprintf(stderr, "Error: input dim (%u) != model dim (%u)\n", orig_dim, model_orig_dim);
        fclose(f);
        return NULL;
    }

    float* proj = (float*)malloc(model_orig_dim * reduced_dim * sizeof(float));
    fread(proj, sizeof(float), model_orig_dim * reduced_dim, f);
    fclose(f);

    float* reduced = (float*)calloc(reduced_dim, sizeof(float));
    for (uint32_t i = 0; i < reduced_dim; i++) {
        for (uint32_t j = 0; j < orig_dim; j++) {
            reduced[i] += vec[j] * proj[j * reduced_dim + i];
        }
    }

    free(proj);
    return reduced;
}

/* ===========================
 * 2️⃣ QUANTISE VECTOR
 * ===========================
 * Quantise a single dim-reduced vector to 5 bits per value.
 * Input:
 *   - vec: pointer to the float vector to quantise
 *   - dim: number of dimensions in the vector
 *   - min_val: global or precomputed min value for this dataset
 *   - max_val: global or precomputed max value for this dataset
 *
 * Output:
 *   - Returns a newly allocated uint8_t array containing the bit-packed 5-bit quantised vector.
 *     You must free() this array after use.
 */
uint8_t* quantise_vector_5bit(const float* vec, uint32_t dim, float min_val, float max_val)
{
    if (!vec || dim == 0) return NULL;

    float range = max_val - min_val;
    if (range <= 1e-9f) range = 1.0f;  // Avoid division by zero

    uint32_t total_bits = dim * BIT_DEPTH;
    uint32_t total_bytes = (total_bits + 7) / 8;

    uint8_t* packed = (uint8_t*)calloc(total_bytes, sizeof(uint8_t));
    if (!packed) {
        fprintf(stderr, "Error: Memory allocation failed in quantise_vector_5bit\n");
        return NULL;
    }

    uint32_t bit_pos = 0;
    for (uint32_t i = 0; i < dim; i++) {
        // Normalise value into [0, 1]
        float normalized = (vec[i] - min_val) / range;
        if (normalized < 0.0f) normalized = 0.0f;
        if (normalized > 1.0f) normalized = 1.0f;

        // Quantise into 5-bit integer
        uint8_t qval = (uint8_t)(normalized * (LEVELS - 1) + 0.5f);

        // Bit-pack into byte array
        for (uint8_t bit = 0; bit < BIT_DEPTH; bit++) {
            if (qval & (1 << bit)) {
                packed[bit_pos / 8] |= (1 << (bit_pos % 8));
            }
            bit_pos++;
        }
    }

    return packed; // caller frees
}

/*
 * Get the top-N clusters that a reduced + quantised query vector belongs to.
 *
 * Inputs:
 *   - reduced_vec: the SVD-reduced float vector (e.g. 100 dims)
 *   - dim: number of dimensions after reduction
 *   - min_val, max_val: normalisation values (same used for quantisation)
 *   - gmm_filename: path to the precomputed .gmm file
 *   - top_k: how many top clusters to return (<= K)
 *
 * Output:
 *   - Returns a malloc’d array of cluster indices sorted by descending probability.
 *     Caller must free() the returned array.
 */
uint32_t* get_top_clusters(const float* reduced_vec, uint32_t dim,
                           float min_val, float max_val,
                           const char* gmm_filename, uint32_t top_k)
{
    if (!reduced_vec || !gmm_filename || top_k == 0) return NULL;

    // -----------------------
    // Step 0: Load GMM file
    // -----------------------
    FILE* f = fopen(gmm_filename, "rb");
    if (!f) {
        perror("Failed to open GMM file");
        return NULL;
    }

    uint32_t gmm_dim;
    float variance;
    float weights[K];
    float means[K][100]; // assumes dim <= 100

    fread(&gmm_dim, sizeof(uint32_t), 1, f);
    fread(&variance, sizeof(float), 1, f);
    fread(weights, sizeof(float), K, f);
    for (uint32_t k = 0; k < K; k++) {
        fread(means[k], sizeof(float), gmm_dim, f);
    }
    fclose(f);

    // Build gmm_params_t struct
    gmm_params_t gmm;
    gmm.dim = gmm_dim;
    gmm.variance = variance;
    memcpy(gmm.weights, weights, sizeof(weights));
    memcpy(gmm.means, means, sizeof(means));

    // -----------------------
    // Step 1: Quantise query vector
    // -----------------------
    uint8_t* qvec = quantise_vector_5bit(reduced_vec, dim, min_val, max_val);
    if (!qvec) {
        fprintf(stderr, "Error: quantisation failed in get_top_clusters\n");
        return NULL;
    }

    // -----------------------
    // Step 2: Dequantise for probability computation
    // -----------------------
    float* dq = (float*)calloc(dim, sizeof(float));
    float range = max_val - min_val;
    if (range <= 1e-9f) range = 1.0f;

    uint32_t bit_pos = 0;
    for (uint32_t d = 0; d < dim; d++) {
        uint8_t qval = 0;
        for (uint8_t b = 0; b < BIT_DEPTH; b++) {
            if (qvec[(bit_pos + b) / 8] & (1 << ((bit_pos + b) % 8))) {
                qval |= (1 << b);
            }
        }
        dq[d] = min_val + (qval / (float)(LEVELS - 1)) * range;
        bit_pos += BIT_DEPTH;
    }
    free(qvec);

    // -----------------------
    // Step 3: Compute cluster probabilities
    // -----------------------
    float probs[K];
    compute_cluster_probabilities(dq, &gmm, probs);
    free(dq);

    // -----------------------
    // Step 4: Select top-K clusters
    // -----------------------
    uint32_t* top_clusters = (uint32_t*)malloc(top_k * sizeof(uint32_t));
    float temp_probs[K];
    memcpy(temp_probs, probs, sizeof(float) * K);

    for (uint32_t i = 0; i < top_k; i++) {
        uint32_t best = 0;
        for (uint32_t j = 1; j < K; j++) {
            if (temp_probs[j] > temp_probs[best]) best = j;
        }
        top_clusters[i] = best;
        temp_probs[best] = -1.0f; // mark as used
    }

    // Optional debug
    printf("Top %u clusters: ", top_k);
    for (uint32_t i = 0; i < top_k; i++) {
        printf("%u (%.4f) ", top_clusters[i], probs[top_clusters[i]]);
    }
    printf("\n");

    return top_clusters;
}


/**
 * Read an .fvecs file into a 2D float array.
 * 
 * Inputs:
 *   - filename: path to the .fvecs file
 *   - out_n: pointer to uint32_t to store the number of vectors
 *   - out_dim: pointer to uint32_t to store the dimension of each vector
 * 
 * Returns:
 *   - malloc'd float** array (each row is a vector). Caller must free each row and the array.
 *   - On error, returns NULL and sets out_n and out_dim to 0.
 */
float** read_fvecs_2d(const char* filename, uint32_t* out_n, uint32_t* out_dim) {
    if (!filename || !out_n || !out_dim) return NULL;

    *out_n = 0;
    *out_dim = 0;

    FILE* f = fopen(filename, "rb");
    if (!f) {
        perror("Failed to open .fvecs file");
        return NULL;
    }

    // Determine file size
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    // Read first vector to get dimension
    uint32_t dim;
    if (fread(&dim, sizeof(uint32_t), 1, f) != 1) {
        perror("Failed to read vector dimension");
        fclose(f);
        return NULL;
    }

    // Compute total number of vectors
    size_t vec_bytes = sizeof(uint32_t) + dim * sizeof(float);
    uint32_t n = fsize / vec_bytes;

    // Allocate array
    float** data = (float**)malloc(n * sizeof(float*));
    if (!data) {
        fprintf(stderr, "Memory allocation failed for read_fvecs\n");
        fclose(f);
        return NULL;
    }

    // Read all vectors
    fseek(f, 0, SEEK_SET);
    for (uint32_t i = 0; i < n; i++) {
        uint32_t d;
        if (fread(&d, sizeof(uint32_t), 1, f) != 1) {
            fprintf(stderr, "Failed to read vector %u dimension\n", i);
            n = i;
            break;
        }
        if (d != dim) {
            fprintf(stderr, "Dimension mismatch at vector %u\n", i);
            n = i;
            break;
        }

        data[i] = (float*)malloc(dim * sizeof(float));
        if (fread(data[i], sizeof(float), dim, f) != dim) {
            fprintf(stderr, "Failed to read vector %u data\n", i);
            n = i;
            break;
        }
    }

    fclose(f);
    *out_n = n;
    *out_dim = dim;
    return data;
}

/**
 * Helper to free the 2D float array returned by read_fvecs.
 */
void free_fvecs(float** data, uint32_t n) {
    if (!data) return;
    for (uint32_t i = 0; i < n; i++) free(data[i]);
    free(data);
}

void compute_cluster_probabilities(const float* query, const gmm_params_t* gmm, float* probs) {
    // log p_k ∝ log w_k - 0.5 * ||x - μ_k||^2 / var
    float logp[K];
    float max_logp = -INFINITY;
    const float var = fmaxf(gmm->variance, 1e-6f); // floor

    for (uint32_t k = 0; k < K; k++) {
        float dist = 0.0f;
        for (uint32_t d = 0; d < gmm->dim; d++) {
            float diff = query[d] - gmm->means[k][d];
            dist += diff * diff;
        }
        float lp = logf(gmm->weights[k] + 1e-12f) - 0.5f * dist / var;
        logp[k] = lp;
        if (lp > max_logp) max_logp = lp;
    }

    // softmax normalisation in log-space
    float sumexp = 0.0f;
    for (uint32_t k = 0; k < K; k++) {
        probs[k] = expf(logp[k] - max_logp);
        sumexp += probs[k];
    }
    float inv = 1.0f / (sumexp + 1e-12f);
    for (uint32_t k = 0; k < K; k++) probs[k] *= inv;
}

/*
void compute_cluster_probabilities(const float* query, const gmm_params_t* gmm, float* probs) {
    float var = gmm->variance;
    float sum_p = 0.0f;
    for (uint32_t k = 0; k < K; k++) {
        float dist = 0.0f;
        for (uint32_t d = 0; d < gmm->dim; d++) {
            float diff = query[d] - gmm->means[k][d];
            dist += diff * diff;
        }
        float p = gmm->weights[k] * expf(-0.5f * dist / var);
        probs[k] = p;
        sum_p += p;
    }
    for (uint32_t k = 0; k < K; k++) probs[k] /= (sum_p + EPSILON);
}
    */