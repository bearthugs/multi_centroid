#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>
#include "gmm_cluster_runner.h"  // for gmm_params_t

#ifdef __cplusplus
extern "C" {
#endif

/* ===========================
 * SVD reduction
 * ===========================
 */
float* reduce_vector_with_svd(const float* vec, uint32_t orig_dim, const char* model_path);

/* ===========================
 * 5-bit quantisation
 * ===========================
 */
uint8_t* quantise_vector_5bit(const float* vec, uint32_t dim, float min_val, float max_val);

/* ===========================
 * Top-K GMM cluster assignment
 * ===========================
 */
uint32_t* get_top_clusters(const float* reduced_vec, uint32_t dim,
                           float min_val, float max_val,
                           const char* gmm_filename, uint32_t top_k);

/* ===========================
 * .fvecs loader
 * ===========================
 */
float** read_fvecs_2d(const char* filename, uint32_t* out_n, uint32_t* out_dim);
void free_fvecs(float** data, uint32_t n);

void compute_cluster_probabilities(const float* query, const gmm_params_t* gmm, float* probs);

#ifdef __cplusplus
}
#endif

#endif // UTILS_H
