#ifndef FAISS_WRAPPER_H
#define FAISS_WRAPPER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* FaissIndex;

// Create different index types
FaissIndex faiss_create_flat_index(int dim);
FaissIndex faiss_create_hnsw_index(int dim, int M);
FaissIndex faiss_create_ivf_flat_index(int dim, int nlist);
FaissIndex faiss_create_ivf_hnsw_index(int dim, int nlist, int M);
FaissIndex faiss_create_ivf_pq_index(int dim, int nlist, int m, int nbits);

// Train IVF / IVF+PQ indexes before adding vectors
void faiss_train_ivf_index(FaissIndex index, float* data, int n, int dim);

// Add vectors
void faiss_add_vectors(FaissIndex index, float* data, int n, int dim);

// Search vectors
void faiss_search_vectors(FaissIndex index, float* queries, int n_query, int dim, int k, int64_t* labels, float* distances);

void faiss_search(FaissIndex index, int n, float* queries, int dim, int k,
                  float* distances, int64_t* labels);

void faiss_hnsw_set_efSearch(FaissIndex index, int ef);

void faiss_hnsw_set_efConstruction(FaissIndex index, int ef);

// Free index memory
void faiss_free_index(FaissIndex index);

#ifdef __cplusplus
}
#endif

#endif // FAISS_WRAPPER_H
