#ifndef QUANT_FUNCTIONS_H
#define QUANT_FUNCTIONS_H

#include <stdint.h>

#define BIT_DEPTH 5
#define LEVELS (1 << BIT_DEPTH)  // 32 levels

typedef struct {
    uint32_t n;      // number of vectors
    uint32_t dim;    // dimension
    float min_val;
    float max_val;
} header_t;

float* read_fvecs(const char* filename, uint32_t* n, uint32_t* dim);
int write_5bit(const char* out_filename, float* data, uint32_t n, uint32_t dim);
uint8_t* read_packed(const char* filename, header_t* header);

#endif // QUANT_FUNCTIONS_H
