/*
 * File:        omp_timing_reduce_max.c
 *
 * Purpose:
 *   Minimal, benchmark-style OpenMP timing example using an explicit
 *   reduction(max: ...) to compute the effective parallel execution time.
 *
 *   This program is a stricter, cleaner counterpart to omp_timing.c:
 *     - no per-thread printing inside the parallel region
 *     - no critical sections
 *     - timing logic expressed entirely via OpenMP reductions
 *
 *   It demonstrates the *canonical* timing pattern for SPMD-style OpenMP code:
 *     1) synchronize threads before starting the timed region
 *     2) measure local elapsed time per thread
 *     3) reduce with max() to obtain the effective time-to-solution
 *
 * Key concepts:
 *   - reduction(max: variable)
 *   - omp_get_wtime() as a wall-clock timer
 *   - Barriers as timing fences
 *   - Why max(thread_time) is the relevant metric
 *
 * OpenMP features used:
 *   Directives:
 *     - parallel
 *     - barrier
 *     - for
 *     - reduction
 *
 *   Runtime library:
 *     - omp_get_max_threads()
 *     - omp_get_wtime()
 *
 * Compilation (GCC / Clang):
 *   gcc -O2 -Wall -Wextra -Wpedantic -g -fopenmp \
 *       omp_timing_reduce_max.c -o omp_timing_reduce_max
 *
 * Execution:
 *   ./omp_timing_reduce_max [N]
 *
 *   Arguments:
 *     N : number of loop iterations (default: 80,000,000)
 *
 * Notes:
 *   - This file is intended for performance measurement and experimentation.
 *   - No I/O occurs inside the timed region.
 *   - Results depend on CPU frequency scaling, OS scheduling, and system load.
 *
 * References:
 *   - OpenMP API Specification (OpenMP ARB): reduction clause, omp_get_wtime
 */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <omp.h>

/* ---------- argument parsing ---------- */

static long long parse_ll_or_default(int argc, char *argv[], int index, long long def)
{
    if (argc <= index) {
        return def;
    }

    errno = 0;
    char *end = NULL;
    long long v = strtoll(argv[index], &end, 10);

    if (errno != 0 || end == argv[index] || *end != '\0' || v <= 0) {
        fprintf(stderr, "Invalid numeric value: '%s'\n", argv[index]);
        fprintf(stderr, "Usage: %s [N]\n", argv[0]);
        exit(1);
    }

    return v;
}

/* ---------- synthetic workload ---------- */

/*
 * burn_cpu():
 * Deterministic floating-point work to consume CPU cycles.
 * Uses volatile to prevent dead-code elimination.
 */
static double burn_cpu(void)
{
    volatile double acc = 0.0;
    for (int i = 0; i < 400; ++i) {
        acc += (double)i * 1e-6;
    }
    return (double)acc;
}

/* ---------- main ---------- */

int main(int argc, char *argv[])
{
    const long long default_n = 80000000LL;
    long long n = parse_ll_or_default(argc, argv, 1, default_n);

    printf("OpenMP timing (reduction(max))\n");
    printf("N = %lld\n", n);
    printf("Max threads available: %d\n\n", omp_get_max_threads());

    double elapsed_max = 0.0;
    double sink = 0.0;

    /*
     * Timing methodology:
     *   - barrier before start: align clocks
     *   - barrier after work: ensure all threads finished
     *   - reduction(max: ...) to compute effective time
     */
    #pragma omp parallel default(none) shared(n) reduction(max:elapsed_max) reduction(+:sink)
    {
        /* Align start time across threads */
        #pragma omp barrier
        double t0 = omp_get_wtime();

        /* Parallel workload */
        #pragma omp for schedule(static)
        for (long long i = 0; i < n; ++i) {
            sink += burn_cpu();
        }

        /* Align end time across threads */
        #pragma omp barrier
        double t1 = omp_get_wtime();

        double local_elapsed = t1 - t0;

        /* Reduction: maximum local elapsed time */
        elapsed_max = local_elapsed;
    }

    printf("Effective parallel time (max thread): %.6f s\n", elapsed_max);

    /*
     * Print sink to ensure computation is observable.
     * The numeric value is not meaningful.
     */
    printf("Computation sink (ignore): %.6f\n\n", sink);

    printf("Interpretation:\n");
    printf("  - Each thread measures its own elapsed time for the same parallel phase.\n");
    printf("  - The effective execution time is the maximum of these local times.\n");
    printf("  - reduction(max: ...) expresses this directly and avoids manual synchronization.\n");
    printf("  - This pattern is suitable for clean performance experiments and scaling studies.\n");

    return 0;
}
