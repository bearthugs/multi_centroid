#include "faiss_wrapper.h"
#include <IndexHNSW.h>
#include <IndexFlat.h>
#include <IndexIVFFlat.h>
#include <index_io.h>
#include <MetaIndexes.h>
#include <cstdio>
#include <vector>
#include <iostream>
#include <Index.h>

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

void faiss_hnsw_set_efSearch(FaissIndex index, int ef) {
    faiss::Index* idx = reinterpret_cast<faiss::Index*>(index);
    faiss::IndexHNSW* hnsw = dynamic_cast<faiss::IndexHNSW*>(idx);
    if (!hnsw) {
        fprintf(stderr, "Error: Index is not an HNSW index\n");
        return;
    }
    hnsw->hnsw.efSearch = ef;
}

void faiss_hnsw_set_efConstruction(FaissIndex index, int ef) {
    faiss::Index* idx = reinterpret_cast<faiss::Index*>(index);
    faiss::IndexHNSW* hnsw = dynamic_cast<faiss::IndexHNSW*>(idx);
    if (!hnsw) {
        fprintf(stderr, "Error: Index is not an HNSW index\n");
        return;
    }
    hnsw->hnsw.efConstruction = ef;
}

// Save HNSW index to file
void faiss_save_index(FaissIndex index, const char* filename) {
    faiss::Index* idx = reinterpret_cast<faiss::Index*>(index);
    faiss::write_index(idx, filename);
}

// Load HNSW index from file
FaissIndex faiss_load_index(const char* filename) {
    faiss::Index* idx = faiss::read_index(filename);
    return reinterpret_cast<FaissIndex>(idx);
}

} // extern "C"