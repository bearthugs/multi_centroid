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

FaissIndexHandle faiss_create_hnsw_index(int d) {
    auto wrapper = new FaissIndexWrapper();
    // Use IndexHNSWFlat with L2 metric
    wrapper->index = new faiss::IndexHNSWFlat(d, 32);
    wrapper->index->hnsw.efConstruction = 40;
    wrapper->index->hnsw.efSearch = 16;
    return (FaissIndexHandle)wrapper;
}

void faiss_add_vectors(FaissIndexHandle handle, int n, float* vectors) {
    FaissIndexWrapper* wrapper = (FaissIndexWrapper*)handle;
    wrapper->index->add(n, vectors);
}

void faiss_search(FaissIndexHandle handle, int nq, float* queries, int k, float* distances, int64_t* labels) {
    FaissIndexWrapper* wrapper = (FaissIndexWrapper*)handle;
    wrapper->index->search(nq, queries, k, distances, labels);
}

void faiss_free_index(FaissIndexHandle handle) {
    FaissIndexWrapper* wrapper = (FaissIndexWrapper*)handle;
    delete wrapper->index;
    delete wrapper;
}

} // extern "C"