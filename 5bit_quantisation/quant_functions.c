#include "quant_functions.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <inttypes.h>


// Reads the .fvecs files
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

    //printf("File: %s, filesize=%zu bytes\n", filename, filesize);
    //printf("vector_size=%zu, computed n=%u\n", vector_size, *n);

    fclose(f);
    return data;
}


/*
PURPOSE: Quantizes floating-point vectors into 5-bit integers per dimension and writes them to a packed binary file.

INPUT: Output filename, original file, number of vectors, dimensions per vector

OUTPUT FILE STRUCTURE:
- A header that contains: number of vectors, dimensionality of each vector,
                          minimum value across all floats, maximum value across all floats
- Followed by all the vectors
*/
int write_5bit(const char* out_filename, float* data, uint32_t n, uint32_t dim) {
    FILE* f = fopen(out_filename, "wb");
    if (!f) { perror("fopen"); return -1; }

    // Find GLOBAL min/max for scaling
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

    // Quantise and bit pack (pack into byte sized units)
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

/*
PURPOSE: Reads the .5bit file

INPUT: The filename and a header struct

OUTPUT: An array of (bit-packed) bytes + updates the header struct
*/
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

/*
PURPOSE:
    Computes the distance between two quantised vectors from a packed 5-bit representation.

PARAMETERS:
    - packed: pointer to packed quantised data
    - idx1, idx2: indices of the two vectors to compare
    - header: header struct containing n, dim, min_val, max_val
    - type: either DIST_EUCLIDEAN or DIST_COSINE

RETURNS:
    Euclidean or cosine distance between the two vectors

Can optimise this function for avx-512
*/
float distance(const uint8_t* packed, uint32_t idx1, uint32_t idx2,
               const header_t* header, distance_type_t type) {
    uint32_t dim = header->dim;
    float min_val = header->min_val;
    float max_val = header->max_val;
    float range = max_val - min_val;

    if (range == 0.0f) return 0.0f; // Avoid division by zero if all values identical

    // Precompute vector starting bit offsets
    uint32_t start_bit_1 = idx1 * dim * BIT_DEPTH;
    uint32_t start_bit_2 = idx2 * dim * BIT_DEPTH;

    float sum_sq = 0.0f;      // For Euclidean
    float dot = 0.0f;         // For cosine
    float norm1 = 0.0f;
    float norm2 = 0.0f;

    for (uint32_t d = 0; d < dim; d++) {
        // --- Unpack 5-bit value for vector 1 ---
        uint32_t bit_pos_1 = start_bit_1 + d * BIT_DEPTH;
        uint8_t q1 = 0;
        for (uint8_t bit = 0; bit < BIT_DEPTH; bit++) {
            uint32_t byte_idx = (bit_pos_1 + bit) / 8;
            uint8_t bit_offset = (bit_pos_1 + bit) % 8;
            if (packed[byte_idx] & (1 << bit_offset))
                q1 |= (1 << bit);
        }

        // --- Unpack 5-bit value for vector 2 ---
        uint32_t bit_pos_2 = start_bit_2 + d * BIT_DEPTH;
        uint8_t q2 = 0;
        for (uint8_t bit = 0; bit < BIT_DEPTH; bit++) {
            uint32_t byte_idx = (bit_pos_2 + bit) / 8;
            uint8_t bit_offset = (bit_pos_2 + bit) % 8;
            if (packed[byte_idx] & (1 << bit_offset))
                q2 |= (1 << bit);
        }

        // --- Dequantise ---
        float v1 = min_val + (range * q1) / (LEVELS - 1);
        float v2 = min_val + (range * q2) / (LEVELS - 1);

        if (type == DIST_EUCLIDEAN) {
            float diff = v1 - v2;
            sum_sq += diff * diff;
        } else if (type == DIST_COSINE) {
            dot += v1 * v2;
            norm1 += v1 * v1;
            norm2 += v2 * v2;
        }
    }

    if (type == DIST_EUCLIDEAN)
        return sqrtf(sum_sq);

    // Handle zero norm cases safely
    float denom = sqrtf(norm1) * sqrtf(norm2);
    if (denom == 0.0f) return 0.0f;

    // Return cosine *distance* (1 - cosine similarity)
    float cosine_similarity = dot / denom;
    return 1.0f - cosine_similarity;
}