/*
 * File:        omp_false_sharing_array.c
 *
 * Purpose:
 *   Demonstrates false sharing and data-layout effects using two common patterns:
 *
 *     1) Array-of-Structs (AoS):   struct { long long a, b; } data[N]
 *     2) Struct-of-Arrays (SoA):   long long a[N], b[N]
 *
 *   Each thread repeatedly updates elements that are "owned" by that thread.
 *   When per-thread elements are stored contiguously and multiple threads update
 *   adjacent elements, false sharing can occur if those elements share a cache line.
 *
 *   This example makes the layout effect visible and also provides a padded AoS
 *   variant to show one mitigation strategy.
 *
 * Key concepts:
 *   - False sharing vs true sharing
 *   - Cache line effects in contiguous arrays
 *   - AoS vs SoA layout and typical performance consequences
 *   - Padding / alignment as a false-sharing mitigation
 *
 * OpenMP features used:
 *   Directives:
 *     - parallel
 *     - for
 *
 *   Runtime library:
 *     - omp_get_max_threads()
 *     - omp_get_num_threads()
 *     - omp_get_thread_num()
 *     - omp_get_wtime()
 *
 * Compilation (GCC/Clang):
 *   gcc -O2 -Wall -Wextra -Wpedantic -g -fopenmp \
 *       omp_false_sharing_array.c -o omp_false_sharing_array
 *
 * Execution:
 *   ./omp_false_sharing_array [elements] [iters] [reps]
 *
 *   Arguments:
 *     elements : number of logical elements per thread (default: 1024)
 *     iters    : updates per element (default: 200000)
 *     reps     : repetitions for each benchmark (default: 5)
 *
 * Examples:
 *   OMP_NUM_THREADS=8 ./omp_false_sharing_array 1024 200000 5
 *   OMP_NUM_THREADS=4 ./omp_false_sharing_array 4096 100000 3
 *
 * Notes:
 *   - This is a microbenchmark; results depend on CPU, cache line size, compiler,
 *     OpenMP runtime, and system load.
 *   - The goal is to compare relative behavior across layouts under contention.
 *   - The padded AoS variant assumes a 64-byte cache line (common), but may not
 *     match all architectures.
 *
 * References:
 *   - OpenMP API Specification (OpenMP ARB): shared-memory execution model
 *   - Compiler docs: optimization and OpenMP runtime behavior
 */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>

/* Didactic cache line assumption */
#define CACHELINE_BYTES 64

/* ---------- argument parsing helpers ---------- */

static long long parse_ll_or_default(int argc, char *argv[], int index, long long def)
{
    if (argc <= index) {
        return def;
    }

    errno = 0;
    char *end = NULL;
    long long v = strtoll(argv[index], &end, 10);

    if (errno != 0 || end == argv[index] || *end != '\0' || v <= 0) {
        fprintf(stderr, "Invalid numeric value at argv[%d]: '%s'\n", index, argv[index]);
        fprintf(stderr, "Usage: %s [elements] [iters] [reps]\n", argv[0]);
        exit(1);
    }

    return v;
}

static int parse_int_or_default(int argc, char *argv[], int index, int def)
{
    if (argc <= index) {
        return def;
    }

    errno = 0;
    char *end = NULL;
    long v = strtol(argv[index], &end, 10);

    if (errno != 0 || end == argv[index] || *end != '\0' || v <= 0) {
        fprintf(stderr, "Invalid integer value at argv[%d]: '%s'\n", index, argv[index]);
        fprintf(stderr, "Usage: %s [elements] [iters] [reps]\n", argv[0]);
        exit(1);
    }

    return (int)v;
}

/* ---------- benchmark data layouts ---------- */

/* AoS element: two counters that are updated together */
typedef struct {
    long long a;
    long long b;
} aos_pair_t;

/*
 * Padded AoS element: force each element into a separate cache line.
 * This is often wasteful in memory, but is useful as a didactic mitigation.
 */
typedef struct {
    long long a;
    long long b;
    unsigned char pad[CACHELINE_BYTES - ((2 * (int)sizeof(long long)) % CACHELINE_BYTES)];
} aos_pair_padded_t;

#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
typedef struct {
    _Alignas(CACHELINE_BYTES) aos_pair_padded_t v;
} aligned_aos_pair_padded_t;
#else
typedef struct {
    aos_pair_padded_t v;
} aligned_aos_pair_padded_t;
#endif

/* ---------- common utility ---------- */

static int get_used_threads(void)
{
    int used = 1;
    #pragma omp parallel default(none) shared(used)
    {
        #pragma omp single
        {
            used = omp_get_num_threads();
        }
    }
    return used;
}

/*
 * Map each thread to a contiguous block of 'elements_per_thread' elements.
 * Thread t owns indices [t*E, (t+1)*E).
 */
static void thread_owned_range(int tid, int elements_per_thread, int *begin, int *end)
{
    *begin = tid * elements_per_thread;
    *end = (tid + 1) * elements_per_thread;
}

/* ---------- Benchmarks ---------- */

/*
 * AoS benchmark:
 * Each thread updates its own contiguous block:
 *   data[idx].a++, data[idx].b++
 *
 * Potential false sharing occurs at boundaries between thread blocks, because:
 *   ... thread t writes element K, thread t+1 writes element K+1 ...
 * If these elements share a cache line, writes can ping-pong the cache line.
 */
static double bench_aos(int elements_per_thread, int iters, int nthreads, long long *checksum)
{
    int n = elements_per_thread * nthreads;

    aos_pair_t *data = (aos_pair_t *)calloc((size_t)n, sizeof(aos_pair_t));
    if (data == NULL) {
        fprintf(stderr, "Allocation failure for AoS array.\n");
        exit(1);
    }

    double t0 = omp_get_wtime();

    #pragma omp parallel default(none) shared(data, elements_per_thread, iters, nthreads)
    {
        int tid = omp_get_thread_num();
        int begin = 0, end = 0;
        thread_owned_range(tid, elements_per_thread, &begin, &end);

        for (int rep = 0; rep < iters; ++rep) {
            for (int i = begin; i < end; ++i) {
                data[i].a += 1;
                data[i].b += 1;
            }
        }
    }

    double t1 = omp_get_wtime();

    long long sum = 0;
    for (int i = 0; i < n; ++i) {
        sum += data[i].a + data[i].b;
    }

    free(data);
    *checksum = sum;
    return t1 - t0;
}

/*
 * Padded AoS benchmark:
 * Same update pattern as AoS, but each element is padded to (approximately) a cache line.
 * This reduces false sharing at thread-block boundaries.
 */
static double bench_aos_padded(int elements_per_thread, int iters, int nthreads, long long *checksum)
{
    int n = elements_per_thread * nthreads;

    aligned_aos_pair_padded_t *data =
        (aligned_aos_pair_padded_t *)calloc((size_t)n, sizeof(aligned_aos_pair_padded_t));
    if (data == NULL) {
        fprintf(stderr, "Allocation failure for padded AoS array.\n");
        exit(1);
    }

    double t0 = omp_get_wtime();

    #pragma omp parallel default(none) shared(data, elements_per_thread, iters, nthreads)
    {
        int tid = omp_get_thread_num();
        int begin = 0, end = 0;
        thread_owned_range(tid, elements_per_thread, &begin, &end);

        for (int rep = 0; rep < iters; ++rep) {
            for (int i = begin; i < end; ++i) {
                data[i].v.a += 1;
                data[i].v.b += 1;
            }
        }
    }

    double t1 = omp_get_wtime();

    long long sum = 0;
    for (int i = 0; i < n; ++i) {
        sum += data[i].v.a + data[i].v.b;
    }

    free(data);
    *checksum = sum;
    return t1 - t0;
}

/*
 * SoA benchmark:
 * Separate arrays:
 *   a[i]++, b[i]++
 *
 * SoA can improve cache behavior depending on access patterns, but false sharing
 * at boundaries can still exist because adjacent indices may share a cache line.
 * The primary didactic point: layout matters and interacts with access patterns.
 */
static double bench_soa(int elements_per_thread, int iters, int nthreads, long long *checksum)
{
    int n = elements_per_thread * nthreads;

    long long *a = (long long *)calloc((size_t)n, sizeof(long long));
    long long *b = (long long *)calloc((size_t)n, sizeof(long long));
    if (a == NULL || b == NULL) {
        fprintf(stderr, "Allocation failure for SoA arrays.\n");
        free(a);
        free(b);
        exit(1);
    }

    double t0 = omp_get_wtime();

    #pragma omp parallel default(none) shared(a, b, elements_per_thread, iters, nthreads)
    {
        int tid = omp_get_thread_num();
        int begin = 0, end = 0;
        thread_owned_range(tid, elements_per_thread, &begin, &end);

        for (int rep = 0; rep < iters; ++rep) {
            for (int i = begin; i < end; ++i) {
                a[i] += 1;
                b[i] += 1;
            }
        }
    }

    double t1 = omp_get_wtime();

    long long sum = 0;
    for (int i = 0; i < n; ++i) {
        sum += a[i] + b[i];
    }

    free(a);
    free(b);

    *checksum = sum;
    return t1 - t0;
}

/* ---------- main ---------- */

int main(int argc, char *argv[])
{
    const int default_elements_per_thread = 1024;
    const int default_iters = 200000;
    const int default_reps = 5;

    int elements_per_thread = parse_int_or_default(argc, argv, 1, default_elements_per_thread);
    int iters = parse_int_or_default(argc, argv, 2, default_iters);
    int reps = parse_int_or_default(argc, argv, 3, default_reps);

    printf("OpenMP false sharing: AoS vs SoA (with padded AoS)\n");
    printf("elements_per_thread = %d, iters = %d, reps = %d\n",
           elements_per_thread, iters, reps);
    printf("Max threads available: %d\n", omp_get_max_threads());
    printf("Assumed cache line size: %d bytes\n\n", CACHELINE_BYTES);

    int nthreads = get_used_threads();
    int n = elements_per_thread * nthreads;

    printf("Threads used: %d\n", nthreads);
    printf("Total elements: %d\n\n", n);

    /*
     * Expected checksum:
     * Each element increments (a and b) by iters, so contribution per element is 2*iters.
     * Total sum = n * 2 * iters.
     */
    long long expected = (long long)n * 2LL * (long long)iters;
    printf("Expected checksum: %lld\n\n", expected);

    double sum_aos = 0.0, sum_soa = 0.0, sum_aos_pad = 0.0;

    for (int r = 1; r <= reps; ++r) {
        long long chk_aos = 0;
        long long chk_soa = 0;
        long long chk_aos_pad = 0;

        double t_aos = bench_aos(elements_per_thread, iters, nthreads, &chk_aos);
        double t_soa = bench_soa(elements_per_thread, iters, nthreads, &chk_soa);
        double t_aos_pad = bench_aos_padded(elements_per_thread, iters, nthreads, &chk_aos_pad);

        sum_aos += t_aos;
        sum_soa += t_soa;
        sum_aos_pad += t_aos_pad;

        printf("Rep %d/%d:\n", r, reps);
        printf("  AoS:        time = %.6f s, checksum = %lld\n", t_aos, chk_aos);
        printf("  SoA:        time = %.6f s, checksum = %lld\n", t_soa, chk_soa);
        printf("  AoS padded: time = %.6f s, checksum = %lld\n", t_aos_pad, chk_aos_pad);

        if (chk_aos != expected || chk_soa != expected || chk_aos_pad != expected) {
            printf("  Warning: checksum mismatch (unexpected; indicates a bug or overflow).\n");
        }

        printf("\n");
    }

    double avg_aos = sum_aos / (double)reps;
    double avg_soa = sum_soa / (double)reps;
    double avg_aos_pad = sum_aos_pad / (double)reps;

    printf("Average timings over %d repetitions:\n", reps);
    printf("  AoS:        %.6f s\n", avg_aos);
    printf("  SoA:        %.6f s\n", avg_soa);
    printf("  AoS padded: %.6f s\n", avg_aos_pad);

    if (avg_aos_pad > 0.0) {
        printf("\nRatios (higher means slower than padded AoS):\n");
        printf("  AoS / AoS_padded: %.2f x\n", avg_aos / avg_aos_pad);
        printf("  SoA / AoS_padded: %.2f x\n", avg_soa / avg_aos_pad);
    }

    printf("\nInterpretation:\n");
    printf("  - If AoS is slower than AoS_padded, boundary false sharing is likely.\n");
    printf("  - SoA vs AoS may differ due to memory layout effects and cache behavior.\n");
    printf("  - Padding is an explicit mitigation but increases memory footprint.\n");
    printf("  - Real codes typically mitigate false sharing by aligning/padding per-thread\n");
    printf("    structures or by changing ownership/blocking to reduce boundary contention.\n");

    return 0;
}
