#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <float.h>
#include <math.h>
#include "utils.h"              // <-- includes your quantized distance functions
#include "multi_centroid_query.h"     // <-- header declaration (see below)

// Maximum clusters allowed to be searched
#define MAX_CLUSTER_SEARCH 100

// -----------------------------
// Data structures
// -----------------------------

typedef struct {
    int id;
    float dist;
} Neighbor;

typedef struct {
    Neighbor *items;
    int size;
    int capacity;
} NeighborList;

// -----------------------------
// Helper functions
// -----------------------------

static NeighborList *create_neighbor_list(int capacity) {
    NeighborList *nl = malloc(sizeof(NeighborList));
    nl->items = malloc(sizeof(Neighbor) * capacity);
    nl->size = 0;
    nl->capacity = capacity;
    return nl;
}

static void free_neighbor_list(NeighborList *nl) {
    free(nl->items);
    free(nl);
}

static void add_neighbor(NeighborList *nl, int id, float dist) {
    if (nl->size < nl->capacity) {
        nl->items[nl->size++] = (Neighbor){id, dist};
    } else {
        // Find worst (farthest) neighbor
        int worst = 0;
        float max_dist = nl->items[0].dist;
        for (int i = 1; i < nl->size; i++) {
            if (nl->items[i].dist > max_dist) {
                worst = i;
                max_dist = nl->items[i].dist;
            }
        }
        if (dist < max_dist) {
            nl->items[worst].id = id;
            nl->items[worst].dist = dist;
        }
    }
}

static int compare_neighbors(const void *a, const void *b) {
    const Neighbor *na = (const Neighbor *)a;
    const Neighbor *nb = (const Neighbor *)b;
    return (na->dist > nb->dist) - (na->dist < nb->dist);
}

// -----------------------------
// Main search function
// -----------------------------

NeighborList *cluster_search_quantized(
    const uint8_t *query,                    // quantized query vector
    int query_dim,
    int num_clusters,
    const float *query_probs,                // length = num_clusters
    const Cluster *clusters,                 // your cluster array
    int k,
    float threshold
) {
    // 1. Identify the top cluster
    int top_cluster = 0;
    float max_prob = -1.0f;
    for (int i = 0; i < num_clusters; i++) {
        if (query_probs[i] > max_prob) {
            max_prob = query_probs[i];
            top_cluster = i;
        }
    }

    NeighborList *neighbors = create_neighbor_list(k);
    bool searched[num_clusters];
    for (int i = 0; i < num_clusters; i++) searched[i] = false;
    searched[top_cluster] = true;

    int searched_count = 1;

    // 2. Search primary cluster
    for (int i = 0; i < clusters[top_cluster].size; i++) {
        float dist = quantized_distance(query, clusters[top_cluster].vectors[i], query_dim);
        add_neighbor(neighbors, clusters[top_cluster].ids[i], dist);
    }

    // 3. Determine secondary clusters based on neighbour probabilities
    bool cluster_to_search[num_clusters];
    for (int i = 0; i < num_clusters; i++) cluster_to_search[i] = false;

    for (int n = 0; n < neighbors->size; n++) {
        int id = neighbors->items[n].id;
        const float *probs = clusters->vector_probs[id]; // You should have this per-vector
        for (int c = 0; c < num_clusters; c++) {
            if (c != top_cluster && probs[c] > threshold)
                cluster_to_search[c] = true;
        }
    }

    // 4. Search secondary clusters (early stopping at 100 clusters total)
    for (int c = 0; c < num_clusters && searched_count < MAX_CLUSTER_SEARCH; c++) {
        if (!cluster_to_search[c] || searched[c]) continue;
        searched[c] = true;
        searched_count++;

        for (int i = 0; i < clusters[c].size; i++) {
            float dist = quantized_distance(query, clusters[c].vectors[i], query_dim);
            add_neighbor(neighbors, clusters[c].ids[i], dist);
        }
    }

    // 5. Sort final neighbor list by distance
    qsort(neighbors->items, neighbors->size, sizeof(Neighbor), compare_neighbors);

    return neighbors;
}
