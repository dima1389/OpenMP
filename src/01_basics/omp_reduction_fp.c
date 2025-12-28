/*
 * File:        omp_reduction_fp.c
 *
 * Purpose:
 *   Intermediate OpenMP example demonstrating floating-point reduction behavior.
 *   The program computes sums in double precision and compares:
 *     1) A serial reference sum
 *     2) A correct parallel reduction sum using reduction(+:sum)
 *
 *   The key observation is that floating-point addition is not associative:
 *     (a + b) + c  may not equal  a + (b + c)
 *   Therefore, even when both computations are correct, the serial and parallel
 *   results may differ slightly because OpenMP reduction combines partial sums
 *   in a different order (a different reduction tree).
 *
 * Key concepts:
 *   - OpenMP reduction on floating-point types
 *   - Floating-point non-associativity and reproducibility implications
 *   - Quantifying differences: absolute error and relative error
 *   - Timing with omp_get_wtime()
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
 * Compilation (GCC/Clang):
 *   gcc -O2 -Wall -Wextra -Wpedantic -g -fopenmp omp_reduction_fp.c -o omp_reduction_fp
 *
 * Execution:
 *   ./omp_reduction_fp [N]
 *
 *   N = number of terms to sum for the series:
 *       sum_{i=1..N} (1.0 / i)
 *   If N is omitted, a reasonable default is used.
 *
 * Notes:
 *   - The parallel reduction is correct, but the final value may differ from the
 *     serial result by a small amount due to different summation order.
 *   - If you require bitwise reproducibility, you typically need:
 *       - a fixed reduction tree,
 *       - deterministic scheduling,
 *       - or compensated summation / reproducible algorithms.
 *     Such approaches are beyond the scope of this intermediate example.
 *
 * References:
 *   - OpenMP API Specification (OpenMP ARB): reduction clause semantics
 *   - Compiler OpenMP runtime documentation (libgomp / libomp)
 *   - IEEE 754 floating-point arithmetic (for background on rounding behavior)
 */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <math.h>
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

/*
 * Serial summation of the harmonic series:
 *   H_n = sum_{i=1..n} (1.0 / i)
 *
 * This is intentionally chosen because:
 * - terms vary in magnitude,
 * - rounding effects accumulate,
 * - and summation order impacts the final result.
 */
static double harmonic_serial(long long n)
{
    double sum = 0.0;
    for (long long i = 1; i <= n; ++i) {
        sum += 1.0 / (double)i;
    }
    return sum;
}

int main(int argc, char *argv[])
{
    /* Default chosen to be large enough for timing and to expose rounding differences. */
    const long long default_n = 200000000LL;
    const long long n = parse_n_or_default(argc, argv, default_n);

    printf("OpenMP floating-point reduction demonstration (intermediate)\n");
    printf("Series: H_N = sum_{i=1..N} (1.0 / i)\n");
    printf("N = %lld\n", n);
    printf("Max threads available (omp_get_max_threads): %d\n\n", omp_get_max_threads());

    /* -------------------- Serial baseline -------------------- */
    double t0 = omp_get_wtime();
    double serial = harmonic_serial(n);
    double t1 = omp_get_wtime();
    double serial_time = t1 - t0;

    printf("[Serial]   H_N = %.17g, time = %.6f s\n", serial, serial_time);

    /* -------------------- Parallel reduction -------------------- */
    /*
     * Use reduction(+:sum) to produce a correct parallel sum.
     * Even though the math is "the same", the combination order differs from serial,
     * so the final floating-point value may not match exactly.
     */
    double reduced = 0.0;

    t0 = omp_get_wtime();

    #pragma omp parallel for default(none) shared(n) reduction(+:reduced) schedule(static)
    for (long long i = 1; i <= n; ++i) {
        reduced += 1.0 / (double)i;
    }

    t1 = omp_get_wtime();
    double reduced_time = t1 - t0;

    printf("[Reduced]  H_N = %.17g, time = %.6f s\n", reduced, reduced_time);

    /* -------------------- Error metrics -------------------- */
    double abs_err = fabs(reduced - serial);

    /*
     * Relative error: abs_err / |serial|
     * For this series serial > 0, but we still write it defensively.
     */
    double rel_err = (serial != 0.0) ? (abs_err / fabs(serial)) : 0.0;

    printf("\nDifference analysis (Reduced vs Serial):\n");
    printf("  Absolute error: %.17g\n", abs_err);
    printf("  Relative error: %.17g\n", rel_err);

    /*
     * A practical interpretation:
     * - If abs_err is small (e.g., around 1e-12 to 1e-9 depending on N and platform),
     *   both are “correct” in a numerical-analysis sense.
     * - The exact magnitude depends on CPU, compiler, optimization flags, and thread count.
     */
    printf("\nInterpretation:\n");
    printf("  - If the results differ slightly, this is expected due to floating-point\n");
    printf("    rounding and different summation order in the reduction tree.\n");
    printf("  - If you require reproducible results, you need a reproducible summation\n");
    printf("    approach (fixed reduction tree or compensated summation).\n");

    /* -------------------- Speedup indicator -------------------- */
    if (reduced_time > 0.0) {
        printf("\nSpeedup (Serial/Reduced): %.2f x\n", serial_time / reduced_time);
    }

    return 0;
}
