#ifndef BENCHMARK_UTILS_H
#define BENCHMARK_UTILS_H

float* load_fvecs(const char* filename, int* out_num_vectors, int* out_dim);
int load_fvecs_both(const char* filename, float** row_major, float** col_major,
                    int* out_num_vectors, int* out_dim);

int* load_ivecs(const char* filename, int* out_num_queries, int* out_k);

double compute_recall_at_k(const int64_t* retrieved, int n_query, int k,
                           const int* gt, int gt_k);

#endif
