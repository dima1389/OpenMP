/*
 * OMP_Parallel_Sum_Comparison.c
 *
 * Compare two OpenMP implementations for summing 1..n:
 *   (A) reduction(+:sum)            (reference / previously solved pattern)
 *   (B) manual partial sums per thread + final accumulation
 *
 * Build:
 *   gcc -O3 -fopenmp -Wall -Wextra OMP_Parallel_Sum_Comparison.c -o OMP_Parallel_Sum_Comparison.exe
 *
 * Run:
 *   OMP_Parallel_Sum_Comparison.exe 1000000000
 *   OMP_Parallel_Sum_Comparison.exe 1000000000 10     (optional repeats; best time is reported)
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

/* Reduce false sharing when multiple threads write partial sums */
#define CACHELINE_BYTES 64
#define PAD_LL (CACHELINE_BYTES / (int)sizeof(long long))

static long long sum_reduction(long long n, double *elapsed_sec)
{
    double t0 = omp_get_wtime();

    long long sum = 0;

    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (long long i = 1; i <= n; ++i) {
        sum += i;
    }

    double t1 = omp_get_wtime();
    *elapsed_sec = t1 - t0;
    return sum;
}

static long long sum_manual_partials(long long n, double *elapsed_sec)
{
    int T = omp_get_max_threads();

    /* Allocate padded per-thread partial sums to minimize cache-line contention */
    long long *partial = (long long*)calloc((size_t)T * PAD_LL, sizeof(long long));
    if (!partial) {
        fprintf(stderr, "Allocation failed.\n");
        exit(1);
    }

    double t0 = omp_get_wtime();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        long long local = 0;

        /*
         * Each thread computes its own partial sum over a disjoint iteration subset.
         * With schedule(static), each thread gets a contiguous block (typical).
         */
        #pragma omp for schedule(static)
        for (long long i = 1; i <= n; ++i) {
            local += i;
        }

        /* Write this thread's partial sum to its own padded slot */
        partial[(size_t)tid * PAD_LL] = local;
    }

    /* Final accumulation of partial sums (serial on the master thread) */
    long long sum = 0;
    for (int t = 0; t < T; ++t) {
        sum += partial[(size_t)t * PAD_LL];
    }

    double t1 = omp_get_wtime();
    *elapsed_sec = t1 - t0;

    free(partial);
    return sum;
}

static long long sum_closed_form(long long n)
{
    /* Sum 1..n = n(n+1)/2; uses 64-bit arithmetic */
    return (n * (n + 1)) / 2;
}

int main(int argc, char **argv)
{
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Usage: %s <n> [repeats]\n", argv[0]);
        return 1;
    }

    long long n = atoll(argv[1]);
    if (n < 0) {
        fprintf(stderr, "Error: n must be >= 0.\n");
        return 1;
    }

    int repeats = 1;
    if (argc == 3) {
        repeats = atoi(argv[2]);
        if (repeats < 1) repeats = 1;
    }

    printf("OpenMP max threads: %d\n", omp_get_max_threads());
    printf("n = %lld, repeats = %d\n\n", n, repeats);

    long long expected = sum_closed_form(n);

    /* Measure best-of-N to reduce noise */
    double best_red = 1e300, best_man = 1e300;
    long long sum_red_best = 0, sum_man_best = 0;

    for (int r = 0; r < repeats; ++r) {
        double t_red = 0.0, t_man = 0.0;

        long long s_red = sum_reduction(n, &t_red);
        long long s_man = sum_manual_partials(n, &t_man);

        if (t_red < best_red) { best_red = t_red; sum_red_best = s_red; }
        if (t_man < best_man) { best_man = t_man; sum_man_best = s_man; }
    }

    /* Basic correctness check */
    int ok_red = (sum_red_best == expected);
    int ok_man = (sum_man_best == expected);

    printf("Expected (closed form) : %lld\n\n", expected);

    printf("[A] reduction(+:sum)\n");
    printf("    sum    : %lld   (%s)\n", sum_red_best, ok_red ? "OK" : "MISMATCH");
    printf("    time   : %.6f s\n\n", best_red);

    printf("[B] manual partial sums + final accumulation\n");
    printf("    sum    : %lld   (%s)\n", sum_man_best, ok_man ? "OK" : "MISMATCH");
    printf("    time   : %.6f s\n\n", best_man);

    if (best_man > 0.0) {
        printf("Speed ratio (A/B): %.3f  (values > 1 mean reduction is slower than manual)\n",
               best_red / best_man);
    }

    return 0;
}
