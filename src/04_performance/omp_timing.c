/*
 * File:        omp_timing.c
 *
 * Purpose:
 *   Demonstrates correct timing methodology for OpenMP programs and explains
 *   why the *maximum per-thread elapsed time* is typically the relevant metric.
 *
 *   This program measures:
 *     1) Per-thread elapsed time for a parallel workload
 *     2) The maximum elapsed time across threads (the effective parallel time)
 *
 *   It also illustrates best practices:
 *     - synchronize threads before starting the timed region
 *     - synchronize threads after finishing the timed region
 *     - avoid timing I/O-heavy code (I/O serializes and distorts performance)
 *
 * Key concepts:
 *   - omp_get_wtime(): wall-clock time measurement
 *   - Barriers as timing fences
 *   - Local (per-thread) timing vs global (program) timing
 *   - Why max(thread_time) approximates end-to-end SPMD time
 *
 * OpenMP features used:
 *   Directives:
 *     - parallel
 *     - barrier
 *     - for
 *     - reduction
 *     - single
 *
 *   Runtime library:
 *     - omp_get_max_threads()
 *     - omp_get_num_threads()
 *     - omp_get_thread_num()
 *     - omp_get_wtime()
 *
 * Compilation (GCC / Clang):
 *   gcc -O2 -Wall -Wextra -Wpedantic -g -fopenmp \
 *       omp_timing.c -o omp_timing
 *
 * Execution:
 *   ./omp_timing [N] [pattern]
 *
 *   Arguments:
 *     N        : number of iterations (default: 80,000,000)
 *     pattern  : workload pattern selector (default: 1)
 *                1 = uniform cost          (similar cost per iteration)
 *                2 = increasing cost       (more expensive as i increases)
 *                3 = periodic spikes       (occasional expensive iterations)
 *
 * Examples:
 *   OMP_NUM_THREADS=8 ./omp_timing 80000000 1
 *   OMP_NUM_THREADS=8 ./omp_timing 80000000 3
 *
 * Notes:
 *   - This is an instructional timing example, not a robust benchmark framework.
 *   - Results depend on CPU frequency scaling, OS scheduling, and background load.
 *
 * References:
 *   - OpenMP API Specification (OpenMP ARB): omp_get_wtime, barrier, reduction
 */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <math.h>
#include <omp.h>

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
        fprintf(stderr, "Usage: %s [N] [pattern]\n", argv[0]);
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

    if (errno != 0 || end == argv[index] || *end != '\0') {
        fprintf(stderr, "Invalid integer value at argv[%d]: '%s'\n", index, argv[index]);
        fprintf(stderr, "Usage: %s [N] [pattern]\n", argv[0]);
        exit(1);
    }

    return (int)v;
}

/* ---------- synthetic workload ---------- */

/*
 * workload_units(i, n, pattern):
 * Returns a small integer representing how much work iteration i performs.
 * This lets us simulate balanced and imbalanced workloads without I/O.
 */
static int workload_units(long long i, long long n, int pattern)
{
    if (pattern == 1) {
        return 8; /* uniform */
    } else if (pattern == 2) {
        /* increasing cost towards the end */
        double x = (double)i / (double)n;
        return 1 + (int)(120.0 * x * x);
    } else {
        /* periodic spikes */
        const long long period = 10000;
        const long long spike_width = 200;
        long long phase = i % period;
        return (phase < spike_width) ? 180 : 3;
    }
}

/*
 * burn_cpu(units):
 * Deterministic floating-point work to consume CPU cycles.
 * Uses volatile to discourage dead-code elimination.
 */
static double burn_cpu(int units)
{
    volatile double acc = 0.0;
    const int inner = 120;

    for (int u = 0; u < units; ++u) {
        for (int k = 0; k < inner; ++k) {
            acc += (double)u * 1e-6 + (double)k * 1e-7;
        }
    }

    return (double)acc;
}

/* ---------- main ---------- */

int main(int argc, char *argv[])
{
    const long long default_n = 80000000LL;
    const int default_pattern = 1;

    long long n = parse_ll_or_default(argc, argv, 1, default_n);
    int pattern = parse_int_or_default(argc, argv, 2, default_pattern);

    if (pattern < 1 || pattern > 3) {
        fprintf(stderr, "Invalid pattern: %d (valid: 1..3)\n", pattern);
        return 1;
    }

    printf("OpenMP timing demonstration\n");
    printf("N = %lld, pattern = %d\n", n, pattern);
    printf("Max threads available: %d\n\n", omp_get_max_threads());

    /*
     * We record:
     *   - local_elapsed: time measured by each thread for the same region
     *   - elapsed_max: maximum across threads (effective parallel time)
     */
    double elapsed_max = 0.0;

    /* Used only as a computation sink to prevent dead-code elimination */
    double global_sink = 0.0;

    /*
     * Timing methodology:
     *   - barrier before start: align start time across threads
     *   - barrier after work: ensure all threads finished before stop
     */
    #pragma omp parallel default(none) shared(n, pattern, elapsed_max, global_sink)
    {
        double local_sink = 0.0;

        /* 1) Synchronize before starting timed section */
        #pragma omp barrier
        double local_start = omp_get_wtime();

        /*
         * Parallel workload: each thread executes its subset of iterations.
         * Using schedule(static) for determinism; other schedules can be tested
         * by modifying this directive or adding a schedule(runtime) variant.
         */
        #pragma omp for schedule(static)
        for (long long i = 0; i < n; ++i) {
            local_sink += burn_cpu(workload_units(i, n, pattern));
        }

        /* 2) Synchronize after finishing the work */
        #pragma omp barrier
        double local_finish = omp_get_wtime();

        double local_elapsed = local_finish - local_start;

        /*
         * Reduce local_elapsed to the maximum across all threads.
         * The maximum is typically the effective time-to-solution because the program
         * cannot complete the parallel phase until the slowest thread finishes.
         */
        #pragma omp critical
        {
            /* keep a sink to discourage overly aggressive optimizations */
            global_sink += local_sink;
        }

        #pragma omp barrier

        #pragma omp single
        {
            elapsed_max = 0.0;
        }

        #pragma omp barrier

        #pragma omp atomic
        global_sink += 0.0; /* no-op to maintain a shared observable */

        #pragma omp barrier

        /*
         * Use a reduction for the maximum elapsed time.
         * Since we are already inside a parallel region, we implement the reduction
         * using a local variable and a critical update for portability and clarity.
         * (A dedicated 'reduction(max: ...)' clause is also possible in OpenMP.)
         */
        #pragma omp critical
        {
            if (local_elapsed > elapsed_max) {
                elapsed_max = local_elapsed;
            }
        }

        /*
         * Optional: print per-thread times (kept minimal; printing perturbs timing).
         * For pedagogical purposes, we print only once per thread.
         */
        #pragma omp barrier
        #pragma omp critical
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            printf("Thread %d/%d local_elapsed = %.6f s\n", tid, nt, local_elapsed);
        }
    }

    printf("\nMax thread elapsed time (effective parallel time): %.6f s\n", elapsed_max);

    /*
     * Print the sink to prevent the compiler from treating the whole workload as dead code.
     * The numeric value has no meaning; it only ensures the computation is observable.
     */
    printf("Computation sink (ignore): %.6f\n\n", global_sink);

    printf("Interpretation:\n");
    printf("  - Each thread measures its own elapsed time for the same parallel phase.\n");
    printf("  - The program's time-to-solution for that phase is bounded by the slowest thread.\n");
    printf("  - Therefore, max(local_elapsed) is the most relevant timing metric in SPMD-style OpenMP.\n");
    printf("  - Barriers around the timed region reduce measurement skew.\n");

    return 0;
}
