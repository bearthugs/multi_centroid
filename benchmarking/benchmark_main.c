#include <stdio.h>

void run_benchmark(const char*, const char*, int);

int main() {
    const char* base_path = "../fvecs_data/sift-128-euclidean_train.fvecs";
    const char* query_path = "../fvecs_data/sift-128-euclidean_test.fvecs";
    int k = 100;

    run_benchmark(base_path, query_path, k);
    return 0;
}
