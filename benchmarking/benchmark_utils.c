#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

// Load the data files
// Returns: float pointer to all vectors (malloc'ed, caller frees)
float* load_fvecs(const char* filename, int* out_num_vectors, int* out_dim) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        perror("Failed to open file");
        return NULL;
    }

    // Read dimension of first vector
    int dim = 0;
    if (fread(&dim, sizeof(int), 1, f) != 1) {
        perror("Failed to read dimension");
        fclose(f);
        return NULL;
    }
    if (dim <= 0 || dim > 10000) {
        fprintf(stderr, "Unreasonable dimension: %d\n", dim);
        fclose(f);
        return NULL;
    }

    // Get file size
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f); // Check ftell
    fseek(f, 0, SEEK_SET);

    // Each vector = 4 bytes (dim) + dim * 4 bytes (floats)
    long vec_size = sizeof(int) + dim * sizeof(float);
    int num_vectors = (int)(file_size / vec_size);

    // Allocate memory for all vectors (just floats, no dimension)
    float* data_row_major = (float*)malloc(num_vectors * dim * sizeof(float)); // Do the same but for column
    if (!data_row_major) {
        perror("malloc failed");
        fclose(f);
        return NULL;
    }

    for (int i = 0; i < num_vectors; i++) {
        int d = 0;
        if (fread(&d, sizeof(int), 1, f) != 1) {
            fprintf(stderr, "Failed to read dim for vector %d\n", i);
            free(data_row_major);
            fclose(f);
            return NULL;
        }
        if (d != dim) {
            fprintf(stderr, "Dimension mismatch at vector %d: expected %d got %d\n", i, dim, d);
            free(data_row_major);
            fclose(f);
            return NULL;
        }
        if (fread(data_row_major + i*dim, sizeof(float), dim, f) != (size_t)dim) {
            fprintf(stderr, "Failed to read vector data for vector %d\n", i);
            free(data_row_major);
            fclose(f);
            return NULL;
        }
    }

    // Create column major

    fclose(f);
    *out_num_vectors = num_vectors;
    *out_dim = dim;
    return data_row_major; // data_column_major
}

int load_fvecs_both(const char* filename, float** row_major, float** col_major, int* out_num_vectors, int* out_dim) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        perror("Failed to open file");
        return -1;
    }

    // Read dimension of first vector
    int dim = 0;
    if (fread(&dim, sizeof(int), 1, f) != 1) {
        perror("Failed to read dimension");
        fclose(f);
        return -1;
    }
    if (dim <= 0 || dim > 10000) {
        fprintf(stderr, "Unreasonable dimension: %d\n", dim);
        fclose(f);
        return -1;
    }

    // Get file size
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f); // Check ftell
    fseek(f, 0, SEEK_SET);

    // Each vector = 4 bytes (dim) + dim * 4 bytes (floats)
    long vec_size = sizeof(int) + dim * sizeof(float);
    int num_vectors = (int)(file_size / vec_size);

    // Allocate memory
    float* data_row = (float*)malloc(num_vectors * dim * sizeof(float));
    float* data_col = (float*)malloc(num_vectors * dim * sizeof(float));
    if (!data_row || !data_col) {
        perror("malloc failed");
        fclose(f);
        free(data_row);
        free(data_col);
        return 0;
    }

    for (int i = 0; i < num_vectors; i++) {
        int d = 0;
        if (fread(&d, sizeof(int), 1, f) != 1 || d != dim) {
            fprintf(stderr, "Dimension mismatch or read failure at vector %d\n", i);
            free(data_row);
            free(data_col);
            fclose(f);
            return 0;
        }

        float* vec = (float*)malloc(dim * sizeof(float));
        if (!vec || fread(vec, sizeof(float), dim, f) != (size_t)dim) {
            fprintf(stderr, "Failed to read vector %d\n", i);
            free(vec);
            free(data_row);
            free(data_col);
            fclose(f);
            return 0;
        }

        // Fill row-major: vector i is at data_row + i*dim
        for (int j = 0; j < dim; j++) {
            data_row[i * dim + j] = vec[j];
            data_col[j * num_vectors + i] = vec[j]; // Column-major: dim-major layout
        }

        free(vec);
    }

    fclose(f);
    *row_major = data_row;
    *col_major = data_col;
    *out_num_vectors = num_vectors;
    *out_dim = dim;
    return 1;
}

// Load ground truth neighbours (int32_t like in fvecs, but IDs instead of floats)
int* load_ivecs(const char* filename, int* out_num_queries, int* out_k) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        perror("Failed to open ivecs file");
        return NULL;
    }

    int dim = 0;
    if (fread(&dim, sizeof(int), 1, f) != 1) {
        perror("Failed to read ivecs dim");
        fclose(f);
        return NULL;
    }

    // Get file size
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    long vec_size = sizeof(int) + dim * sizeof(int);
    int num_vecs = (int)(file_size / vec_size);

    int* data = (int*)malloc(num_vecs * dim * sizeof(int));
    if (!data) {
        perror("malloc failed");
        fclose(f);
        return NULL;
    }

    for (int i = 0; i < num_vecs; i++) {
        int d = 0;
        fread(&d, sizeof(int), 1, f);
        if (d != dim) {
            fprintf(stderr, "Dimension mismatch in ivecs\n");
            free(data);
            fclose(f);
            return NULL;
        }
        fread(data + i * dim, sizeof(int), dim, f);
    }

    fclose(f);
    *out_num_queries = num_vecs;
    *out_k = dim;
    return data;
}

// Compute Recall@k
double compute_recall_at_k(const int64_t* retrieved, int n_query, int k,
                           const int* gt, int gt_k) {
    int correct = 0;
    for (int i = 0; i < n_query; i++) {
        for (int j = 0; j < k; j++) {
            int64_t id = retrieved[i * k + j];
            // check if this id is in the ground truth top-gt_k list
            for (int g = 0; g < gt_k; g++) {
                if (gt[i * gt_k + g] == id) {
                    correct++;
                    break;
                }
            }
        }
    }
    return (double)correct / (n_query * k);
}
