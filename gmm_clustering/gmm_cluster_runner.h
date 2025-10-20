#ifndef GMM_CLUSTER_RUNNER_H
#define GMM_CLUSTER_RUNNER_H

#include "../quant_functions.h"

#define K 8
#define MAX_ITER 50
#define EPSILON 1e-6

// GMM parameter struct for query-time
typedef struct {
    uint32_t dim;
    float means[K][100]; // assume 100-dim reduction target
    float weights[K];
    float variance;
} gmm_params_t;

// Compute posterior probabilities of query belonging to each cluster
void compute_cluster_probabilities(const float* query, const gmm_params_t* gmm, float* probs);

#endif
