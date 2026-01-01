/*
 * File:        omp_reduction_fp_pairwise.c
 *
 * Purpose:
 *   Demonstrates a deterministic, tree-based (pairwise) summation strategy for
 *   floating-point series in OpenMP. This is a follow-up to:
 *     - omp_reduction_fp.c  (naive reduction; correct but not reproducible)
 *     - omp_reduction_fp_compensated.c (improved accuracy; still not bitwise stable)
 *
 *   Here we implement a reproducible approach by enforcing a fixed summation order:
 *     1) Each thread computes a local partial sum over a deterministic static chunk.
 *     2) The partial sums are combined using a fixed binary-tree reduction on one thread.
 *
 *   This improves reproducibility across runs with the same number of threads and
 *   the same scheduling. It also provides a stepping stone to discussing:
 *     - reproducibility requirements in scientific computing,
 *     - and the trade-off between determinism and scalability.
 *
 * Key concepts:
 *   - Floating-point non-associativity
 *   - Deterministic reduction trees (pairwise summation)
 *   - Static partitioning for repeatable work assignment
 *   - Accuracy vs performance vs reproducibility trade-offs
 *
 * OpenMP features used:
 *   Directives:
 *     - parallel
 *     - for
 *     - single
 *     - barrier
 *
 *   Runtime library:
 *     - omp_get_max_threads()
 *     - omp_get_num_threads()
 *     - omp_get_thread_num()
 *     - omp_get_wtime()
 *
 * Compilation (GCC/Clang):
 *   gcc -O2 -Wall -Wextra -Wpedantic -g -fopenmp \
 *       omp_reduction_fp_pairwise.c -o omp_reduction_fp_pairwise
 *
 * Execution:
 *   ./omp_reduction_fp_pairwise [N]
 *
 *   N = number of terms for the harmonic series:
 *       H_N = sum_{i=1..N} (1.0 / i)
 *
 * Notes:
 *   - Determinism/reproducibility properties:
 *       * For a fixed thread count and static scheduling, results should be stable
 *         across runs on the same platform/compiler settings.
 *       * Changing the number of threads changes the partitioning and therefore
 *         changes the result (still correct, but not identical).
 *   - This example uses static partitioning deliberately; dynamic scheduling would
 *     generally destroy reproducibility.
 *
 * References:
 *   - OpenMP API Specification (OpenMP ARB): work-sharing, barriers, single construct
 *   - IEEE 754 floating-point arithmetic (rounding and non-associativity)
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

/* Serial naive summation for comparison. */
static double harmonic_serial_naive(long long n)
{
    double sum = 0.0;
    for (long long i = 1; i <= n; ++i) {
        sum += 1.0 / (double)i;
    }
    return sum;
}

/*
 * Pairwise (tree) reduction of an array of partial sums.
 * The order of operations is deterministic given a fixed array size.
 *
 * Note:
 *   This function mutates the input array 'a' in-place. That is intentional:
 *   we treat 'partials' as scratch space.
 */
static double pairwise_tree_reduce(double *a, int count)
{
    int active = count;

    while (active > 1) {
        int half = active / 2;

        /* Combine pairs: a[k] = a[k] + a[k + half] */
        for (int k = 0; k < half; ++k) {
            a[k] = a[k] + a[k + half];
        }

        /* If odd number of active elements, carry last element forward. */
        if (active % 2 == 1) {
            a[half] = a[active - 1];
            active = half + 1;
        } else {
            active = half;
        }
    }

    return (count > 0) ? a[0] : 0.0;
}

int main(int argc, char *argv[])
{
    const long long default_n = 200000000LL;
    const long long n = parse_n_or_default(argc, argv, default_n);

    printf("OpenMP deterministic pairwise reduction (floating point)\n");
    printf("Series: H_N = sum_{i=1..N} (1.0 / i)\n");
    printf("N = %lld\n", n);
    printf("Max threads available: %d\n\n", omp_get_max_threads());

    /* -------------------- Serial baseline -------------------- */
    double t0 = omp_get_wtime();
    double serial = harmonic_serial_naive(n);
    double t1 = omp_get_wtime();
    double time_serial = t1 - t0;

    printf("[Serial naive] H_N = %.17g, time = %.6f s\n", serial, time_serial);

    /* -------------------- Parallel deterministic partition + pairwise combine -------------------- */
    double parallel_pairwise = 0.0;
    double time_parallel_pairwise = 0.0;

    /*
     * We compute one partial sum per thread, store into an array, then do
     * a deterministic pairwise reduction on a single thread.
     *
     * This enforces a stable final combination order for a fixed thread count.
     */
    t0 = omp_get_wtime();

    int used_threads = 0;
    double *partials = NULL;

    /*
     * IMPORTANT:
     *   We use default(none), so every variable referenced inside must be explicitly
     *   scoped as shared/private/etc.
     */
    #pragma omp parallel default(none) shared(n, used_threads, partials, parallel_pairwise)
    {
        int tid = omp_get_thread_num();

        /*
         * Allocate the partial sums array exactly once (single thread),
         * sized by the actual number of threads used in this parallel region.
         */
        #pragma omp single
        {
            used_threads = omp_get_num_threads();
            partials = (double *)calloc((size_t)used_threads, sizeof(double));
            if (partials == NULL) {
                fprintf(stderr, "Allocation failure for partial sums.\n");
                exit(1);
            }
        }

        /*
         * Ensure all threads see 'partials' and 'used_threads' initialized
         * before writing their own partial sum.
         */
        #pragma omp barrier

        /*
         * Deterministic static partitioning:
         * Each thread sums indices in a reproducible pattern.
         *
         * We intentionally use schedule(static) and avoid reduction(+:sum)
         * to control the reduction tree explicitly.
         */
        double local_sum = 0.0;

        #pragma omp for schedule(static)
        for (long long i = 1; i <= n; ++i) {
            local_sum += 1.0 / (double)i;
        }

        partials[tid] = local_sum;

        /*
         * Ensure all partial sums are written before the final combination step.
         */
        #pragma omp barrier

        /*
         * Combine on a single thread to enforce deterministic order.
         * (The tree order is fixed for a fixed 'used_threads'.)
         */
        #pragma omp single
        {
            parallel_pairwise = pairwise_tree_reduce(partials, used_threads);
            free(partials);
            partials = NULL;
        }
    }

    t1 = omp_get_wtime();
    time_parallel_pairwise = t1 - t0;

    printf("[Pairwise]    H_N = %.17g, time = %.6f s (threads = %d)\n",
           parallel_pairwise, time_parallel_pairwise, used_threads);

    /* -------------------- Error metrics -------------------- */
    double abs_err = fabs(parallel_pairwise - serial);
    double rel_err = (serial != 0.0) ? (abs_err / fabs(serial)) : 0.0;

    printf("\nDifference analysis (Pairwise vs Serial naive):\n");
    printf("  Absolute error: %.17g\n", abs_err);
    printf("  Relative error: %.17g\n", rel_err);

    printf("\nInterpretation:\n");
    printf("  - Pairwise reduction enforces a deterministic combination order of thread partial sums.\n");
    printf("  - This improves run-to-run stability for a fixed thread count and static scheduling.\n");
    printf("  - Changing the thread count changes partitioning and therefore changes the result.\n");
    printf("  - Deterministic strategies can reduce nondeterminism but may reduce scalability.\n");

    if (time_parallel_pairwise > 0.0) {
        printf("\nSpeedup (Serial/Pairwise): %.2f x\n", time_serial / time_parallel_pairwise);
    }

    return 0;
}
