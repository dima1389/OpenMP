/*
 * File:        omp_reduction.c
 *
 * Purpose:
 *   Demonstrates correct accumulation (summation) in OpenMP using the reduction clause.
 *   The program contrasts:
 *     1) A serial reference implementation
 *     2) An intentionally incorrect parallel accumulation (data race) for illustration
 *     3) A correct parallel accumulation using OpenMP reduction(+:sum)
 *
 * Key concepts:
 *   - Race condition on shared variables in parallel regions
 *   - OpenMP reduction clause: per-thread partial results + deterministic combine step
 *   - Work-sharing with parallel for
 *   - Timing with omp_get_wtime()
 *   - Integer overflow considerations in summations
 *
 * OpenMP features used:
 *   Directives:
 *     - parallel for
 *     - reduction
 *
 *   Runtime library:
 *     - omp_get_max_threads()
 *     - omp_get_num_threads()
 *     - omp_get_wtime()
 *
 * Compilation (GCC/Clang):
 *   gcc -O2 -Wall -Wextra -Wpedantic -g -fopenmp omp_reduction.c -o omp_reduction
 *
 * Execution:
 *   ./omp_reduction [N]
 *
 *   N = number of terms to sum, summing i = 1..N.
 *   If N is omitted, a reasonable default is used.
 *
 * Notes:
 *   - The "incorrect parallel" variant contains a deliberate data race and may produce
 *     different results across runs. It is included for pedagogical contrast only.
 *   - This example uses 64-bit signed integers (long long). For very large N, the sum
 *     may still overflow; see the printed overflow threshold notes.
 *
 * References:
 *   - OpenMP API Specification (OpenMP ARB): reduction clause semantics
 *   - Compiler OpenMP runtime documentation (libgomp / libomp)
 */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <limits.h>
#include <omp.h>

/* Parse a positive long long from argv with basic validation. */
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

/* Serial reference sum: sum_{i=1..n} i */
static long long sum_serial(long long n)
{
    long long sum = 0;
    for (long long i = 1; i <= n; ++i) {
        sum += i;
    }
    return sum;
}

int main(int argc, char *argv[])
{
    /* Default chosen to be large enough to measure time but not too slow. */
    const long long default_n = 100000000LL;
    const long long n = parse_n_or_default(argc, argv, default_n);

    printf("OpenMP reduction demonstration\n");
    printf("N = %lld (summing i = 1..N)\n", n);
    printf("Max threads available (omp_get_max_threads): %d\n\n", omp_get_max_threads());

    /*
     * Overflow note:
     * The exact sum is n(n+1)/2. For signed 64-bit (LLONG_MAX),
     * overflow occurs when n(n+1)/2 > LLONG_MAX.
     * We do not compute the boundary here, but we warn that extremely large N will overflow.
     */
    if (n > 3000000000LL) {
        printf("Warning: N is very large; the 64-bit sum may overflow.\n\n");
    }

    /* -------------------- Serial baseline -------------------- */
    double t0 = omp_get_wtime();
    long long serial = sum_serial(n);
    double t1 = omp_get_wtime();
    double serial_time = t1 - t0;

    printf("[Serial]   sum = %lld, time = %.6f s\n", serial, serial_time);

    /* -------------------- Incorrect parallel version (race) -------------------- */
    /*
     * This is intentionally wrong: sum is shared and updated concurrently without
     * synchronization, producing a data race and undefined behavior.
     */
    long long raced = 0;
    t0 = omp_get_wtime();

    #pragma omp parallel for default(none) shared(n, raced)
    for (long long i = 1; i <= n; ++i) {
        raced += i; /* DATA RACE: incorrect on purpose */
    }

    t1 = omp_get_wtime();
    double raced_time = t1 - t0;

    printf("[Raced]    sum = %lld, time = %.6f s  (expected to be wrong / unstable)\n",
           raced, raced_time);

    /* -------------------- Correct parallel reduction -------------------- */
    long long reduced = 0;
    t0 = omp_get_wtime();

    #pragma omp parallel for default(none) shared(n) reduction(+:reduced)
    for (long long i = 1; i <= n; ++i) {
        reduced += i;
    }

    t1 = omp_get_wtime();
    double reduced_time = t1 - t0;

    printf("[Reduced]  sum = %lld, time = %.6f s\n", reduced, reduced_time);

    /* -------------------- Validation -------------------- */
    /*
     * If overflow did not occur, serial and reduced should match exactly.
     * The raced version is not expected to match.
     */
    if (reduced == serial) {
        printf("\nValidation: PASS (reduction matches serial reference)\n");
    } else {
        printf("\nValidation: FAIL (reduction does not match serial reference)\n");
        printf("Possible causes:\n");
        printf("  - Integer overflow for large N\n");
        printf("  - Nonstandard compiler/runtime behavior (unlikely)\n");
    }

    /*
     * Optional: print a simple speedup indicator relative to serial.
     * Note: For small N, overhead may dominate and parallel may be slower.
     */
    if (reduced_time > 0.0) {
        printf("Speedup (Serial/Reduced): %.2f x\n", serial_time / reduced_time);
    }

    return 0;
}
