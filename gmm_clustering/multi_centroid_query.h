#ifndef CLUSTER_SEARCH_H
#define CLUSTER_SEARCH_H

#include <stdint.h>

typedef struct {
    const uint8_t **vectors;     // array of quantized vectors
    const int *ids;              // original IDs
    int size;
    const float **vector_probs;  // probability of each vector belonging to clusters
} Cluster;

typedef struct {
    int id;
    float dist;
} Neighbor;

typedef struct {
    Neighbor *items;
    int size;
    int capacity;
} NeighborList;

// Main search function
NeighborList *cluster_search_quantized(
    const uint8_t *query,
    int query_dim,
    int num_clusters,
    const float *query_probs,
    const Cluster *clusters,
    int k,
    float threshold
);

#endif
