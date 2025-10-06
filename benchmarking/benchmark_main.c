#include <stdio.h>

void run_benchmark_hnsw(const char*, const char*, const char*, int, int);

int main() {
    int k = 100;

    const char* base_path = "../fvecs_data/coco-i2i-512-angular_train.fvecs";
    const char* query_path = "../fvecs_data/coco-i2i-512-angular_test.fvecs";
    const char* neighbour_path = "../fvecs_data/coco-i2i-512-angular_neighbours.ivecs";
    printf("Dataset: coco-i2i-512-angular\n");
    run_benchmark_hnsw(base_path, query_path, neighbour_path, k, 1);
    printf("\n\n\n\n\n");

    base_path = "../fvecs_data/deep-image-96-angular_train.fvecs";
    query_path = "../fvecs_data/deep-image-96-angular_test.fvecs";
    neighbour_path = "../fvecs_data/deep-image-96-angular_neighbours.ivecs";
    //printf("Dataset: deep-image-96-angular\n");
    //run_benchmark_hnsw(base_path, query_path, neighbour_path, k, 1);
    //printf("\n\n\n\n\n");

    base_path = "../fvecs_data/fashion-mnist-784-euclidean_train.fvecs";
    query_path = "../fvecs_data/fashion-mnist-784-euclidean_test.fvecs";
    neighbour_path = "../fvecs_data/fashion-mnist-784-euclidean_neighbours.ivecs";
    printf("Dataset: fashion-mnist-784-euclidean\n");
    run_benchmark_hnsw(base_path, query_path, neighbour_path, k, 0);
    printf("\n\n\n\n\n");

    
    base_path = "../fvecs_data/gist-960-euclidean_train.fvecs";
    query_path = "../fvecs_data/gist-960-euclidean_test.fvecs";
    neighbour_path = "../fvecs_data/gist-960-euclidean_neighbours.ivecs";
    printf("Dataset: gist-960-euclidean\n");
    run_benchmark_hnsw(base_path, query_path, neighbour_path, k, 0);
    printf("\n\n\n\n\n");

    base_path = "../fvecs_data/glove-25-angular_train.fvecs";
    query_path = "../fvecs_data/glove-25-angular_test.fvecs";
    neighbour_path = "../fvecs_data/glove-25-angular_neighbours.ivecs";
    printf("Dataset: glove-25-angular\n");
    run_benchmark_hnsw(base_path, query_path, neighbour_path, k, 1);
    printf("\n\n\n\n\n");

    base_path = "../fvecs_data/glove-50-angular_train.fvecs";
    query_path = "../fvecs_data/glove-50-angular_test.fvecs";
    neighbour_path = "../fvecs_data/glove-50-angular_neighbours.ivecs";
    printf("Dataset: glove-50-angular\n");
    run_benchmark_hnsw(base_path, query_path, neighbour_path, k, 1);
    printf("\n\n\n\n\n");

    base_path = "../fvecs_data/glove-100-angular_train.fvecs";
    query_path = "../fvecs_data/glove-100-angular_test.fvecs";
    neighbour_path = "../fvecs_data/glove-100-angular_neighbours.ivecs";
    printf("Dataset: glove-100-angular\n");
    run_benchmark_hnsw(base_path, query_path, neighbour_path, k, 1);
    printf("\n\n\n\n\n");

    base_path = "../fvecs_data/glove-200-angular_train.fvecs";
    query_path = "../fvecs_data/glove-200-angular_test.fvecs";
    neighbour_path = "../fvecs_data/glove-200-angular_neighbours.ivecs";
    printf("Dataset: glove-200-angular\n");
    run_benchmark_hnsw(base_path, query_path, neighbour_path, k, 1);
    printf("\n\n\n\n\n");

    base_path = "../fvecs_data/lastfm-64-dot_train.fvecs";
    query_path = "../fvecs_data/lastfm-64-dot_test.fvecs";
    neighbour_path = "../fvecs_data/lastfm-64-dot_neighbours.ivecs";
    printf("Dataset: lastfm-64-dot\n");
    run_benchmark_hnsw(base_path, query_path, neighbour_path, k, 1);
    printf("\n\n\n\n\n");

    base_path = "../fvecs_data/mnist-784-euclidean_train.fvecs";
    query_path = "../fvecs_data/mnist-784-euclidean_test.fvecs";
    neighbour_path = "../fvecs_data/mnist-784-euclidean_neighbours.ivecs";
    printf("Dataset: mnist-784-euclidean\n");
    run_benchmark_hnsw(base_path, query_path, neighbour_path, k, 0);
    printf("\n\n\n\n\n");

    base_path = "../fvecs_data/nytimes-256-angular_train.fvecs";
    query_path = "../fvecs_data/nytimes-256-angular_test.fvecs";
    neighbour_path = "../fvecs_data/nytimes-256-angular_neighbours.ivecs";
    printf("Dataset: nytimes-256-angular\n");
    run_benchmark_hnsw(base_path, query_path, neighbour_path, k, 1);
    printf("\n\n\n\n\n");

    base_path = "../fvecs_data/sift-128-euclidean_train.fvecs";
    query_path = "../fvecs_data/sift-128-euclidean_test.fvecs";
    neighbour_path = "../fvecs_data/sift-128-euclidean_neighbours.ivecs";
    printf("Dataset: sift-128-euclidean\n");
    run_benchmark_hnsw(base_path, query_path, neighbour_path, k, 0);
    printf("\n\n\n\n\n");
    
    return 0;
}
