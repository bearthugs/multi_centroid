// compare_ivecs.c
// ------------------------------------------------------------
// Compare ANN results (.ivecs) against ground-truth neighbours.
//
// Computes Recall@100:
//   recall = average_i |P[i] ∩ G[i]| / 100
//
// COMPILE:
//   gcc -O2 gmm_querying/compare_ivecs.c -o compare_ivecs
/*
// RUN:
   ./compare_ivecs \
        querying_results/K128/coco-i2i-512-angular_test_reduced_gmm_candidates.ivecs \
        fvecs_data/coco-i2i-512-angular_neighbours.ivecs

    ./compare_ivecs \
        querying_results/sift-128-euclidean_test_reduced_gmm_candidates.ivecs \
        fvecs_data/sift-128-euclidean_neighbours.ivecs

    ./compare_ivecs \
        querying_results/K128/glove-25-angular_test_reduced_gmm_candidates.ivecs \
        fvecs_data/glove-25-angular_neighbours.ivecs

    ./compare_ivecs \
        querying_results/K128/glove-50-angular_test_reduced_gmm_candidates.ivecs \
        fvecs_data/glove-50-angular_neighbours.ivecs
*/
// ------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

static int read_ivecs_row(FILE* f, int* buf, int expected_dim) {
    int dim;
    if (fread(&dim, sizeof(int32_t), 1, f) != 1)
        return 0; // EOF

    if (dim != expected_dim) {
        fprintf(stderr, "Error: ivecs row dimension mismatch (got %d, expected %d)\n",
                dim, expected_dim);
        return -1;
    }

    if (fread(buf, sizeof(int32_t), dim, f) != (size_t)dim) {
        fprintf(stderr, "Error: could not read ivecs ids\n");
        return -1;
    }

    return 1;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr,
            "Usage: %s <predicted.ivecs> <groundtruth.ivecs>\n",
            argv[0]);
        return 1;
    }

    const char* pred_path = argv[1];
    const char* gt_path   = argv[2];

    printf("Predicted:   %s\n", pred_path);
    printf("Groundtruth: %s\n", gt_path);

    FILE* fp = fopen(pred_path, "rb");
    if (!fp) {
        fprintf(stderr, "Failed to open predicted ivecs\n");
        return 1;
    }

    FILE* fg = fopen(gt_path, "rb");
    if (!fg) {
        fprintf(stderr, "Failed to open groundtruth ivecs\n");
        fclose(fp);
        return 1;
    }

    // ==== Read first row to get dimension ====
    int dim_pred;
    if (fread(&dim_pred, sizeof(int32_t), 1, fp) != 1) {
        fprintf(stderr, "Failed to read first row of predicted ivecs\n");
        fclose(fp); fclose(fg);
        return 1;
    }
    fseek(fp, -4, SEEK_CUR); // rewind

    int dim_gt;
    if (fread(&dim_gt, sizeof(int32_t), 1, fg) != 1) {
        fprintf(stderr, "Failed to read first row of groundtruth ivecs\n");
        fclose(fp); fclose(fg);
        return 1;
    }
    fseek(fg, -4, SEEK_CUR);

    if (dim_pred != dim_gt) {
        fprintf(stderr, "Dimension mismatch: pred=%d gt=%d\n",
                dim_pred, dim_gt);
        fclose(fp); fclose(fg);
        return 1;
    }

    int K = dim_pred;  // number of neighbours (should be 100)
    printf("Neighbour count per query = %d\n", K);

    int* pred_buf = (int*)malloc(sizeof(int) * K);
    int*  gt_buf  = (int*)malloc(sizeof(int) * K);

    if (!pred_buf || !gt_buf) {
        fprintf(stderr, "OOM\n");
        fclose(fp); fclose(fg);
        free(pred_buf); free(gt_buf);
        return 1;
    }

    long long shared_total = 0;
    long long total_queries = 0;

    while (1) {
        int ret_p = read_ivecs_row(fp, pred_buf, K);
        int ret_g = read_ivecs_row(fg,  gt_buf,  K);

        if (ret_p == 0 && ret_g == 0) break; // both EOF
        if (ret_p <= 0 || ret_g <= 0) {
            fprintf(stderr, "File length mismatch or read error.\n");
            break;
        }

        // compute shared neighbours
        int shared = 0;
        for (int i = 0; i < K; i++) {
            int pid = pred_buf[i];
            for (int j = 0; j < K; j++) {
                if (gt_buf[j] == pid) {
                    shared++;
                    break;
                }
            }
        }

        shared_total += shared;
        total_queries++;

        if (total_queries % 1000 == 0) {
            printf("Processed %lld queries...\n", total_queries);
        }
    }

    fclose(fp);
    fclose(fg);

    if (total_queries == 0) {
        fprintf(stderr, "No queries found.\n");
    } else {
        double recall = (double)shared_total / (double)(total_queries * K);
        printf("======================================\n");
        printf("Total queries: %lld\n", total_queries);
        printf("Average shared neighbours: %.4f / %d\n",
               (double)shared_total / (double)total_queries, K);
        printf("Recall@%d = %.6f\n", K, recall);
        printf("======================================\n");
    }

    free(pred_buf);
    free(gt_buf);

    return 0;
}
