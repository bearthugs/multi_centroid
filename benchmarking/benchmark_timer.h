#ifndef BENCHMARK_TIMER_H
#define BENCHMARK_TIMER_H

#include <time.h>

static inline double wall_time() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

#endif