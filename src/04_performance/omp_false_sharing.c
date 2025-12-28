/*
 * File:        omp_false_sharing.c
 *
 * Purpose:
 *   Demonstrates the performance impact of false sharing in shared-memory programs.
 *
 *   False sharing occurs when multiple threads frequently write to different variables
 *   that reside on the same CPU cache line. Even though the variables are logically
 *   independent, cache coherence traffic forces repeated invalidation/transfer of the
 *   shared cache line, significantly degrading performance.
 *
 *   This program compares two cases:
 *     1) Packed counters: per-thread counters stored contiguously (high false sharing risk)
 *     2) Padded counters: each per-thread counter placed in a separate cache line
 *
 * Key concepts:
 *   - Cache lines and coherence effects
 *   - False sharing vs true sharing
 *   - Per-thread data placement as a performance optimization
 *   - Microbenchmark methodology: minimize confounders and validate correctness
 *
 * OpenMP features used:
 *   Directives:
 *     - parallel
 *
 *   Runtime library:
 *     - omp_get_max_threads()
 *     - omp_get_num_threads()
 *     - omp_get_thread_num()
 *     - omp_get_wtime()
 *
 * Compilation (GCC/Clang):
 *   gcc -O2 -Wall -Wextra -Wpedantic -g -fopenmp omp_false_sharing.c -o omp_false_sharing
 *
 * Execution:
 *   ./omp_false_sharing [iters] [reps]
 *
 *   Arguments:
 *     iters : number of increments per thread in each repetition (default: 200,000,000)
 *     reps  : number of repetitions for each case (default: 5)
 *
 * Examples:
 *   OMP_NUM_THREADS=8 ./omp_false_sharing 200000000 5
 *   OMP_NUM_THREADS=4 ./omp_false_sharing 500000000 3
 *
 * Notes:
 *   - This is a microbenchmark. Results depend on CPU, cache line size, compiler,
 *     OpenMP runtime, CPU frequency scaling, and system load.
 *   - The padding assumes a 64-byte cache line (common on modern x86_64 and many ARM
 *     systems), but cache line size may differ. This is a didactic assumption.
 *   - Use large iteration counts to make the effect measurable.
 *
 * References:
 *   - OpenMP API Specification (OpenMP ARB): memory model context (shared memory)
 *   - Compiler documentation: optimization and OpenMP runtime behavior
 */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <stdint.h>
#include <omp.h>

/* Parse a positive long long from argv with validation. */
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
        fprintf(stderr, "Usage: %s [iters] [reps]\n", argv[0]);
        exit(1);
    }

    return v;
}

/* Parse a positive int from argv with validation. */
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
        fprintf(stderr, "Usage: %s [iters] [reps]\n", argv[0]);
        exit(1);
    }

    return (int)v;
}

/*
 * Cache line size assumption (didactic default).
 * Many systems use 64 bytes; adjust if needed.
 */
#define CACHELINE_BYTES 64

/*
 * Case 1: Packed counters.
 * - Each thread writes to counters[tid].
 * - Adjacent counters likely share cache lines -> false sharing under contention.
 */
static double bench_packed(long long iters, int nthreads, long long *out_total)
{
    long long *counters = (long long *)calloc((size_t)nthreads, sizeof(long long));
    if (counters == NULL) {
        fprintf(stderr, "Allocation failure for packed counters.\n");
        exit(1);
    }

    double t0 = omp_get_wtime();

    #pragma omp parallel default(none) shared(counters, iters, nthreads)
    {
        int tid = omp_get_thread_num();
        long long *my = &counters[tid];

        /* tight increment loop */
        for (long long i = 0; i < iters; ++i) {
            *my += 1;
        }
    }

    double t1 = omp_get_wtime();

    long long total = 0;
    for (int t = 0; t < nthreads; ++t) {
        total += counters[t];
    }

    free(counters);
    *out_total = total;

    return t1 - t0;
}

/*
 * Case 2: Padded counters.
 * - Each thread writes to its own counter placed in a distinct cache line.
 * - Greatly reduces (often eliminates) false sharing for per-thread updates.
 *
 * Implementation:
 *   Each element is a struct aligned to CACHELINE_BYTES and padded so that
 *   successive elements are CACHELINE_BYTES apart (or more).
 */
typedef struct {
    long long value;
    unsigned char pad[CACHELINE_BYTES - (sizeof(long long) % CACHELINE_BYTES)];
} padded_counter_t;

#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
    /* C11 alignment */
    typedef struct {
        _Alignas(CACHELINE_BYTES) padded_counter_t c;
    } aligned_padded_counter_t;
#else
    /* If C11 alignas is unavailable, we still keep padding; alignment may be weaker. */
    typedef struct {
        padded_counter_t c;
    } aligned_padded_counter_t;
#endif

static double bench_padded(long long iters, int nthreads, long long *out_total)
{
    aligned_padded_counter_t *counters =
        (aligned_padded_counter_t *)calloc((size_t)nthreads, sizeof(aligned_padded_counter_t));

    if (counters == NULL) {
        fprintf(stderr, "Allocation failure for padded counters.\n");
        exit(1);
    }

    double t0 = omp_get_wtime();

    #pragma omp parallel default(none) shared(counters, iters, nthreads)
    {
        int tid = omp_get_thread_num();
        long long *my = &counters[tid].c.value;

        for (long long i = 0; i < iters; ++i) {
            *my += 1;
        }
    }

    double t1 = omp_get_wtime();

    long long total = 0;
    for (int t = 0; t < nthreads; ++t) {
        total += counters[t].c.value;
    }

    free(counters);
    *out_total = total;

    return t1 - t0;
}

int main(int argc, char *argv[])
{
    const long long default_iters = 200000000LL;
    const int default_reps = 5;

    long long iters = parse_ll_or_default(argc, argv, 1, default_iters);
    int reps = parse_int_or_default(argc, argv, 2, default_reps);

    printf("OpenMP false sharing demonstration\n");
    printf("iters per thread = %lld, repetitions = %d\n", iters, reps);
    printf("Max threads available: %d\n", omp_get_max_threads());
    printf("Assumed cache line size: %d bytes\n\n", CACHELINE_BYTES);

    /*
     * Determine how many threads are actually used.
     * We create a tiny parallel region to query omp_get_num_threads().
     */
    int used_threads = 1;
    #pragma omp parallel default(none) shared(used_threads)
    {
        #pragma omp single
        {
            used_threads = omp_get_num_threads();
        }
    }

    printf("Threads used in parallel regions: %d\n\n", used_threads);

    double packed_sum = 0.0;
    double padded_sum = 0.0;

    long long expected_total = iters * (long long)used_threads;

    printf("Expected total increments: %lld\n\n", expected_total);

    /* --- Run packed vs padded multiple times to reduce noise --- */
    for (int r = 1; r <= reps; ++r) {
        long long total_packed = 0;
        long long total_padded = 0;

        double t_packed = bench_packed(iters, used_threads, &total_packed);
        double t_padded = bench_padded(iters, used_threads, &total_padded);

        packed_sum += t_packed;
        padded_sum += t_padded;

        printf("Rep %d/%d:\n", r, reps);
        printf("  Packed: time = %.6f s, total = %lld\n", t_packed, total_packed);
        printf("  Padded: time = %.6f s, total = %lld\n", t_padded, total_padded);

        if (total_packed != expected_total || total_padded != expected_total) {
            printf("  Warning: unexpected total. (Possible overflow or logic issue.)\n");
        }

        printf("\n");
    }

    double packed_avg = packed_sum / (double)reps;
    double padded_avg = padded_sum / (double)reps;

    printf("Average timings over %d repetitions:\n", reps);
    printf("  Packed: %.6f s\n", packed_avg);
    printf("  Padded: %.6f s\n", padded_avg);

    if (padded_avg > 0.0) {
        printf("  Packed/Padded ratio: %.2f x\n", packed_avg / padded_avg);
    }

    printf("\nInterpretation:\n");
    printf("  - If Packed is significantly slower than Padded, false sharing is likely.\n");
    printf("  - Padding places each thread's frequently-written counter in a different cache line,\n");
    printf("    reducing coherence traffic and improving throughput.\n");
    printf("  - Real applications often experience false sharing in arrays of structs, per-thread\n");
    printf("    statistics, and frequently-updated counters.\n");

    return 0;
}
