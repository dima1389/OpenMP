/*
 * File:        omp_atomic_vs_critical.c
 *
 * Purpose:
 *   Demonstrates the semantic and performance differences between:
 *     - #pragma omp atomic
 *     - #pragma omp critical
 *
 *   The program increments a shared counter many times in parallel and measures
 *   elapsed time for:
 *     1) atomic increment
 *     2) critical section increment
 *     3) reduction-based accumulation (included as a best-practice baseline)
 *
 * Key concepts:
 *   - Data races on shared variables
 *   - Mutual exclusion vs atomic read-modify-write
 *   - Overhead trade-offs (atomic typically cheaper than critical for simple updates)
 *   - Reduction as a scalable alternative for associative/commutative operations
 *
 * OpenMP features used:
 *   Directives:
 *     - parallel
 *     - for
 *     - atomic
 *     - critical
 *     - reduction
 *
 *   Runtime library:
 *     - omp_get_max_threads()
 *     - omp_get_wtime()
 *
 * Compilation (GCC/Clang):
 *   gcc -O2 -Wall -Wextra -Wpedantic -g -fopenmp \
 *       omp_atomic_vs_critical.c -o omp_atomic_vs_critical
 *
 * Execution:
 *   ./omp_atomic_vs_critical [N]
 *
 *   N = total number of increments (default: 200,000,000)
 *
 * Notes:
 *   - Results depend on CPU architecture, OpenMP runtime, and thread count.
 *   - This benchmark intentionally creates contention to highlight differences.
 *   - For very large N, runtime may be substantial; adjust as needed.
 *
 * References:
 *   - OpenMP API Specification (OpenMP ARB): atomic, critical, reduction constructs
 */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <omp.h>

/* Parse a positive long long from argv with validation. */
static long long parse_n_or_default(int argc, char *argv[], long long default_n)
{
    if (argc < 2) {
        return default_n;
    }

    errno = 0;
    char *end = NULL;
    long long n = strtoll(argv[1], &end, 10);

    if (errno != 0 || end == argv[1] || *end != '\0' || n <= 0) {
        fprintf(stderr, "Invalid N: '%s'\n", argv[1]);
        fprintf(stderr, "Usage: %s [N]\n", argv[0]);
        exit(1);
    }

    return n;
}

/*
 * Run atomic increment benchmark.
 * Uses omp atomic for a single shared counter update.
 */
static double bench_atomic(long long n, long long *out_counter)
{
    long long counter = 0;

    double t0 = omp_get_wtime();

    #pragma omp parallel for default(none) shared(n, counter)
    for (long long i = 0; i < n; ++i) {
        #pragma omp atomic
        counter += 1;
    }

    double t1 = omp_get_wtime();

    *out_counter = counter;
    return t1 - t0;
}

/*
 * Run critical section increment benchmark.
 * Uses omp critical to serialize the increment.
 */
static double bench_critical(long long n, long long *out_counter)
{
    long long counter = 0;

    double t0 = omp_get_wtime();

    #pragma omp parallel for default(none) shared(n, counter)
    for (long long i = 0; i < n; ++i) {
        #pragma omp critical
        {
            counter += 1;
        }
    }

    double t1 = omp_get_wtime();

    *out_counter = counter;
    return t1 - t0;
}

/*
 * Run reduction benchmark.
 * Each thread increments a private partial counter; OpenMP combines them at the end.
 */
static double bench_reduction(long long n, long long *out_counter)
{
    long long counter = 0;

    double t0 = omp_get_wtime();

    #pragma omp parallel for default(none) shared(n) reduction(+:counter)
    for (long long i = 0; i < n; ++i) {
        counter += 1;
    }

    double t1 = omp_get_wtime();

    *out_counter = counter;
    return t1 - t0;
}

int main(int argc, char *argv[])
{
    const long long default_n = 200000000LL;
    long long n = parse_n_or_default(argc, argv, default_n);

    printf("OpenMP atomic vs critical (with reduction baseline)\n");
    printf("Total increments N = %lld\n", n);
    printf("Max threads available: %d\n\n", omp_get_max_threads());

    long long c_atomic = 0;
    long long c_critical = 0;
    long long c_reduction = 0;

    double t_atomic = bench_atomic(n, &c_atomic);
    double t_critical = bench_critical(n, &c_critical);
    double t_reduction = bench_reduction(n, &c_reduction);

    printf("Results:\n");
    printf("  atomic:    counter=%lld, time=%.6f s\n", c_atomic, t_atomic);
    printf("  critical:  counter=%lld, time=%.6f s\n", c_critical, t_critical);
    printf("  reduction: counter=%lld, time=%.6f s\n\n", c_reduction, t_reduction);

    printf("Validation:\n");
    if (c_atomic == n && c_critical == n && c_reduction == n) {
        printf("  PASS: all methods produced the expected result.\n\n");
    } else {
        printf("  FAIL: unexpected counter value(s).\n");
        printf("  Expected: %lld\n", n);
        printf("  atomic=%lld, critical=%lld, reduction=%lld\n\n",
               c_atomic, c_critical, c_reduction);
    }

    printf("Interpretation:\n");
    printf("  - atomic protects a single read-modify-write update and is usually cheaper\n");
    printf("    than critical for simple operations such as counter increments.\n");
    printf("  - critical provides mutual exclusion for an arbitrary code block, but can\n");
    printf("    impose higher overhead and serialization.\n");
    printf("  - reduction is often the best choice for associative/commutative operations\n");
    printf("    because it minimizes contention (per-thread partial results).\n");

    return 0;
}
