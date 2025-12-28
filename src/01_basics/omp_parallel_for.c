/*
 * File:        omp_parallel_for.c
 *
 * Purpose:
 *   Introduces the OpenMP work-sharing construct: #pragma omp parallel for.
 *
 *   This example demonstrates how a canonical counted loop can be parallelized
 *   using a single OpenMP directive that:
 *     - creates a parallel region
 *     - distributes loop iterations among threads
 *     - synchronizes threads at the end of the loop (implicit barrier)
 *
 *   The program computes the sum of the first N natural numbers in parallel and
 *   contrasts the parallel result with the closed-form serial formula.
 *
 * Key concepts:
 *   - Work-sharing vs explicit parallel regions
 *   - Loop iteration partitioning
 *   - Implicit barriers at the end of work-sharing constructs
 *   - Reduction for safe accumulation
 *
 * OpenMP features used:
 *   Directives:
 *     - parallel for
 *     - reduction
 *
 *   Runtime library:
 *     - omp_get_max_threads()
 *     - omp_get_wtime()
 *
 * Compilation (GCC / Clang):
 *   gcc -O2 -Wall -Wextra -Wpedantic -g -fopenmp \
 *       omp_parallel_for.c -o omp_parallel_for
 *
 * Execution:
 *   ./omp_parallel_for [N]
 *
 *   Arguments:
 *     N : upper bound of summation (default: 100000000)
 *
 * Notes:
 *   - The default scheduling strategy is implementation-defined (often static).
 *   - The reduction clause is required to avoid data races on the shared sum.
 *   - This example emphasizes correctness and semantics rather than performance tuning.
 *
 * References:
 *   - OpenMP API Specification (OpenMP ARB): parallel for, reduction
 */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <omp.h>

/* Parse a positive long long from argv with validation. */
static long long parse_ll_or_default(int argc, char *argv[], int index, long long default_value)
{
    if (argc <= index) {
        return default_value;
    }

    errno = 0;
    char *end = NULL;
    long long v = strtoll(argv[index], &end, 10);

    if (errno != 0 || end == argv[index] || *end != '\0' || v <= 0) {
        fprintf(stderr, "Invalid numeric value at argv[%d]: '%s'\n", index, argv[index]);
        fprintf(stderr, "Usage: %s [N]\n", argv[0]);
        exit(1);
    }

    return v;
}

int main(int argc, char *argv[])
{
    const long long default_n = 100000000LL;
    long long n = parse_ll_or_default(argc, argv, 1, default_n);

    printf("OpenMP parallel for example\n");
    printf("N = %lld\n", n);
    printf("Max threads available: %d\n\n", omp_get_max_threads());

    /*
     * Parallel summation using a reduction.
     * The parallel for directive combines:
     *   - creation of a parallel region
     *   - work-sharing across threads
     */
    long long sum_parallel = 0;

    double t0 = omp_get_wtime();

    #pragma omp parallel for default(none) shared(n) reduction(+:sum_parallel)
    for (long long i = 1; i <= n; ++i) {
        sum_parallel += i;
    }

    double t1 = omp_get_wtime();

    /*
     * Closed-form serial result:
     *   sum_{i=1}^{n} i = n * (n + 1) / 2
     */
    long long sum_serial = n * (n + 1) / 2;

    printf("Parallel sum   = %lld\n", sum_parallel);
    printf("Serial formula = %lld\n", sum_serial);

    if (sum_parallel == sum_serial) {
        printf("Result check: PASS\n");
    } else {
        printf("Result check: FAIL\n");
    }

    printf("Elapsed time (parallel loop): %.6f s\n\n", t1 - t0);

    printf("Interpretation:\n");
    printf("  - #pragma omp parallel for is the most common OpenMP construct for data-parallel loops.\n");
    printf("  - Iterations are divided among threads automatically by the runtime.\n");
    printf("  - The reduction clause safely combines partial results without explicit synchronization.\n");
    printf("  - An implicit barrier occurs at the end of the loop unless 'nowait' is specified.\n");

    return 0;
}
