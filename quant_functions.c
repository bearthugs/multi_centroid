#include "quant_functions.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

float* read_fvecs(const char* filename, uint32_t* n, uint32_t* dim) {
    FILE* f = fopen(filename, "rb");
    if (!f) { perror("fopen"); return NULL; }

    fseek(f, 0, SEEK_END);
    size_t filesize = ftell(f);
    rewind(f);

    uint32_t d;
    fread(&d, sizeof(uint32_t), 1, f);
    *dim = d;

    size_t vector_size = sizeof(uint32_t) + sizeof(float) * d;
    *n = filesize / vector_size;

    float* data = (float*)malloc(*n * d * sizeof(float));
    if (!data) { fclose(f); return NULL; }

    for (uint32_t i = 0; i < *n; i++) {
        fread(&d, sizeof(uint32_t), 1, f);  // dim header
        fread(&data[i * (*dim)], sizeof(float), *dim, f);
    }
    fclose(f);
    return data;
}

int write_5bit(const char* out_filename, float* data, uint32_t n, uint32_t dim) {
    FILE* f = fopen(out_filename, "wb");
    if (!f) { perror("fopen"); return -1; }

    // Find min/max for scaling
    float min_val = data[0], max_val = data[0];
    for (uint32_t i = 0; i < n * dim; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }

    header_t header = { n, dim, min_val, max_val };
    fwrite(&header, sizeof(header_t), 1, f);

    uint32_t total_bits = n * dim * BIT_DEPTH;
    uint32_t total_bytes = (total_bits + 7) / 8;
    uint8_t* packed = (uint8_t*)calloc(total_bytes, sizeof(uint8_t));

    // Quantise and bit pack
    uint32_t bit_pos = 0;
    for (uint32_t i = 0; i < n * dim; i++) {
        float normalized = (data[i] - min_val) / (max_val - min_val);
        uint8_t qval = (uint8_t)(normalized * (LEVELS - 1) + 0.5f);
        for (uint8_t bit = 0; bit < BIT_DEPTH; bit++) {
            if (qval & (1 << bit)) {
                packed[bit_pos / 8] |= (1 << (bit_pos % 8));
            }
            bit_pos++;
        }
    }

    fwrite(packed, sizeof(uint8_t), total_bytes, f);
    free(packed);
    fclose(f);

    return 0;
}

uint8_t* read_packed(const char* filename, header_t* header) {
    FILE* f = fopen(filename, "rb");
    if (!f) { perror("fopen"); return NULL; }

    fread(header, sizeof(header_t), 1, f);

    uint32_t total_bits = header->n * header->dim * BIT_DEPTH;
    uint32_t total_bytes = (total_bits + 7) / 8;

    uint8_t* packed = (uint8_t*)malloc(total_bytes);
    fread(packed, sizeof(uint8_t), total_bytes, f);
    fclose(f);

    return packed;
}
