/*
 * File:        omp_reduction_fp_compensated.c
 *
 * Purpose:
 *   Advanced follow-up to omp_reduction_fp.c demonstrating compensated summation
 *   techniques for improving numerical accuracy in floating-point reductions.
 *
 *   This program compares:
 *     1) Serial naive summation
 *     2) Serial Kahan compensated summation
 *     3) Parallel naive OpenMP reduction
 *     4) Parallel compensated summation using per-thread Kahan accumulation
 *
 *   The example illustrates the trade-offs between:
 *     - numerical accuracy,
 *     - parallel scalability,
 *     - and algorithmic complexity.
 *
 * Key concepts:
 *   - Floating-point rounding error accumulation
 *   - Kahan compensated summation
 *   - Per-thread local accumulation in OpenMP
 *   - Correct combination of partial results
 *   - Accuracy vs performance trade-offs
 *
 * OpenMP features used:
 *   Directives:
 *     - parallel
 *
 *   Runtime library:
 *     - omp_get_max_threads()
 *     - omp_get_thread_num()
 *     - omp_get_wtime()
 *
 * Compilation (GCC/Clang):
 *   gcc -O2 -Wall -Wextra -Wpedantic -g -fopenmp \
 *       omp_reduction_fp_compensated.c -o omp_reduction_fp_compensated
 *
 * Execution:
 *   ./omp_reduction_fp_compensated [N]
 *
 *   N = number of terms for the harmonic series:
 *       H_N = sum_{i=1..N} (1.0 / i)
 *
 * Notes:
 *   - The compensated parallel version is NOT bitwise reproducible across different
 *     thread counts, but it is typically much closer to the serial Kahan reference
 *     than the naive reduction.
 *   - True reproducibility requires a fixed reduction tree and deterministic scheduling.
 *
 * References:
 *   - OpenMP API Specification (OpenMP ARB): reduction semantics
 *   - IEEE 754 Floating-Point Standard
 *   - W. Kahan, "Further remarks on reducing truncation errors"
 */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <math.h>
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

/* Naive serial summation of the harmonic series. */
static double harmonic_serial_naive(long long n)
{
    double sum = 0.0;
    for (long long i = 1; i <= n; ++i) {
        sum += 1.0 / (double)i;
    }
    return sum;
}

/* Serial Kahan compensated summation of the harmonic series. */
static double harmonic_serial_kahan(long long n)
{
    double sum = 0.0;
    double c = 0.0;  /* Compensation for lost low-order bits */

    for (long long i = 1; i <= n; ++i) {
        double y = (1.0 / (double)i) - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}

int main(int argc, char *argv[])
{
    const long long default_n = 200000000LL;
    const long long n = parse_n_or_default(argc, argv, default_n);

    printf("OpenMP compensated floating-point reduction (advanced)\n");
    printf("Series: H_N = sum_{i=1..N} (1.0 / i)\n");
    printf("N = %lld\n", n);
    printf("Max threads available: %d\n\n", omp_get_max_threads());

    /* -------------------- Serial naive -------------------- */
    double t0 = omp_get_wtime();
    double serial_naive = harmonic_serial_naive(n);
    double t1 = omp_get_wtime();
    double time_serial_naive = t1 - t0;

    /* -------------------- Serial Kahan -------------------- */
    t0 = omp_get_wtime();
    double serial_kahan = harmonic_serial_kahan(n);
    t1 = omp_get_wtime();
    double time_serial_kahan = t1 - t0;

    /* -------------------- Parallel naive reduction -------------------- */
    double parallel_naive = 0.0;

    t0 = omp_get_wtime();
    #pragma omp parallel for default(none) shared(n) reduction(+:parallel_naive)
    for (long long i = 1; i <= n; ++i) {
        parallel_naive += 1.0 / (double)i;
    }
    t1 = omp_get_wtime();
    double time_parallel_naive = t1 - t0;

    /* -------------------- Parallel compensated summation -------------------- */
    /*
     * Strategy:
     * - Each thread performs a local Kahan summation over its chunk
     * - Local sums are combined at the end using a critical section
     *
     * This reduces error compared to naive reduction while remaining portable.
     */
    double parallel_kahan = 0.0;

    t0 = omp_get_wtime();

    #pragma omp parallel default(none) shared(n, parallel_kahan)
    {
        double local_sum = 0.0;
        double local_c = 0.0;

        #pragma omp for schedule(static)
        for (long long i = 1; i <= n; ++i) {
            double y = (1.0 / (double)i) - local_c;
            double t = local_sum + y;
            local_c = (t - local_sum) - y;
            local_sum = t;
        }

        #pragma omp critical
        {
            parallel_kahan += local_sum;
        }
    }

    t1 = omp_get_wtime();
    double time_parallel_kahan = t1 - t0;

    /* -------------------- Results -------------------- */
    printf("Results:\n");
    printf("  Serial naive:   %.17g  (%.6f s)\n", serial_naive, time_serial_naive);
    printf("  Serial Kahan:   %.17g  (%.6f s)\n", serial_kahan, time_serial_kahan);
    printf("  Parallel naive: %.17g  (%.6f s)\n", parallel_naive, time_parallel_naive);
    printf("  Parallel Kahan: %.17g  (%.6f s)\n\n", parallel_kahan, time_parallel_kahan);

    /* -------------------- Error analysis -------------------- */
    printf("Absolute error vs serial Kahan (reference):\n");
    printf("  Serial naive:   %.17g\n", fabs(serial_naive - serial_kahan));
    printf("  Parallel naive: %.17g\n", fabs(parallel_naive - serial_kahan));
    printf("  Parallel Kahan: %.17g\n", fabs(parallel_kahan - serial_kahan));

    printf("\nInterpretation:\n");
    printf("  - Serial Kahan provides a high-accuracy reference.\n");
    printf("  - Parallel naive reduction accumulates more rounding error.\n");
    printf("  - Parallel Kahan significantly reduces error at the cost of\n");
    printf("    additional arithmetic and a critical section.\n");
    printf("  - Performance vs accuracy trade-offs must be evaluated per application.\n");

    return 0;
}
