#include "faiss_wrapper.h"
#include <IndexHNSW.h>
#include <IndexFlat.h>
#include <IndexIVFFlat.h>
#include <index_io.h>
#include <MetaIndexes.h>
#include <vector>
#include <iostream>

struct FaissIndexWrapper {
    faiss::IndexHNSWFlat* index;
};

extern "C" {

FaissIndex faiss_create_hnsw_index(int dim, int M) {
    auto wrapper = new faiss::IndexHNSWFlat(dim, M);
    return (FaissIndex)wrapper;
}

void faiss_add_vectors(FaissIndex index, float* data, int n, int dim) {
    auto hnsw_index = (faiss::IndexHNSWFlat*)index;
    hnsw_index->add(n, data);
}

void faiss_search(FaissIndex index, int n, float* queries, int dim, int k,
                  float* distances, int64_t* labels) {
    auto hnsw_index = (faiss::IndexHNSWFlat*)index;
    hnsw_index->search(n, queries, k, distances, labels);
}

void faiss_free_index(FaissIndex index) {
    auto hnsw_index = (faiss::IndexHNSWFlat*)index;
    delete hnsw_index;
}

} // extern "C"