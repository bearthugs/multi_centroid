#ifndef QUANT_FUNCTIONS_H
#define QUANT_FUNCTIONS_H

#include <stdint.h>

#define BIT_DEPTH 5
#define LEVELS (1 << BIT_DEPTH)  // 32 levels

// Header struct for each dataset
typedef struct {
    uint32_t n;      // number of vectors
    uint32_t dim;    // dimension
    float min_val;   // min across all values
    float max_val;   // max across all values
} header_t;

// Type of distance to compute
typedef enum {
    DIST_EUCLIDEAN,
    DIST_COSINE
} distance_type_t;

float* read_fvecs(const char* filename, uint32_t* n, uint32_t* dim);
int write_5bit(const char* out_filename, float* data, uint32_t n, uint32_t dim);
uint8_t* read_packed(const char* filename, header_t* header);
float distance(const uint8_t* packed, uint32_t idx1, uint32_t idx2,
               const header_t* header, distance_type_t type);

#endif // QUANT_FUNCTIONS_H
