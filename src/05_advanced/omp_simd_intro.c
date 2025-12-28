/*
 * File:        omp_simd_intro.c
 *
 * Purpose:
 *   Introduces OpenMP SIMD vectorization directives and demonstrates how they can
 *   complement thread-level parallelism.
 *
 *   This program compares three variants of a simple vector kernel:
 *     1) Serial scalar loop                          (baseline)
 *     2) SIMD vectorized loop via #pragma omp simd   (single-thread SIMD)
 *     3) Combined parallel + SIMD via parallel for simd (thread-level + SIMD)
 *
 *   The kernel computed is a SAXPY-like operation:
 *     y[i] = a * x[i] + y[i]
 *
 * Key concepts:
 *   - SIMD (Single Instruction, Multiple Data) vs multithreading
 *   - #pragma omp simd to request/guide vectorization
 *   - Data alignment and contiguous access patterns
 *   - Reduction for checksum validation
 *   - Benchmark methodology: avoid I/O in timed regions
 *
 * OpenMP features used:
 *   Directives:
 *     - simd
 *     - parallel for simd
 *     - reduction
 *
 *   Runtime library:
 *     - omp_get_max_threads()
 *     - omp_get_wtime()
 *
 * Compilation (GCC / Clang):
 *   gcc -O3 -march=native -ffast-math -Wall -Wextra -Wpedantic -g -fopenmp \
 *       omp_simd_intro.c -o omp_simd_intro
 *
 *   (More conservative flags are possible; -O3 is recommended to enable vectorization.)
 *
 * Execution:
 *   ./omp_simd_intro [N] [reps]
 *
 *   Arguments:
 *     N    : vector length (default: 50,000,000)
 *     reps : repetitions of the kernel (default: 5)
 *
 * Examples:
 *   ./omp_simd_intro 50000000 5
 *   OMP_NUM_THREADS=8 ./omp_simd_intro 50000000 5
 *
 * Notes:
 *   - #pragma omp simd is a *hint/contract* to the compiler; the compiler may still
 *     refuse vectorization if dependencies cannot be proven safe.
 *   - For meaningful SIMD speedups, compile with optimization enabled (-O2/-O3).
 *   - On some systems, -march=native enables use of wider SIMD instructions.
 *
 * References:
 *   - OpenMP API Specification (OpenMP ARB): SIMD construct, parallel for simd
 *   - GCC documentation: "Vector Extensions" and auto-vectorization options
 *   - LLVM/Clang documentation: loop vectorizer and OpenMP SIMD support
 */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
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
        fprintf(stderr, "Usage: %s [N] [reps]\n", argv[0]);
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

    if (errno != 0 || end == argv[index] || *end != '\0' || v <= 0) {
        fprintf(stderr, "Invalid integer value at argv[%d]: '%s'\n", index, argv[index]);
        fprintf(stderr, "Usage: %s [N] [reps]\n", argv[0]);
        exit(1);
    }

    return (int)v;
}

/* ---------- memory utilities ---------- */

/*
 * Allocate an array of doubles with a requested alignment (best-effort).
 * This helps demonstrate typical alignment practices for SIMD.
 *
 * Uses C11 aligned_alloc where available; falls back to malloc otherwise.
 * Note: aligned_alloc requires size be a multiple of alignment.
 */
static void *aligned_alloc_best_effort(size_t alignment, size_t size)
{
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
    size_t rounded = (size + alignment - 1u) / alignment * alignment;
    void *p = aligned_alloc(alignment, rounded);
    if (p == NULL) {
        fprintf(stderr, "aligned_alloc failed (size=%zu, alignment=%zu)\n", rounded, alignment);
        exit(1);
    }
    return p;
#else
    (void)alignment;
    void *p = malloc(size);
    if (p == NULL) {
        fprintf(stderr, "malloc failed (size=%zu)\n", size);
        exit(1);
    }
    return p;
#endif
}

/* ---------- initialization and checks ---------- */

static void init_vectors(double *x, double *y, long long n)
{
    for (long long i = 0; i < n; ++i) {
        x[i] = 1.0 + (double)(i % 100) * 0.001;
        y[i] = 2.0 - (double)(i % 100) * 0.001;
    }
}

/* Compute a checksum to validate that computations are not optimized away. */
static double checksum(const double *y, long long n)
{
    double s = 0.0;

    #pragma omp simd reduction(+:s)
    for (long long i = 0; i < n; ++i) {
        s += y[i];
    }

    return s;
}

/* ---------- kernels ---------- */

static double run_serial(double *x, double *y, long long n, int reps, double a)
{
    double t0 = omp_get_wtime();

    for (int r = 0; r < reps; ++r) {
        for (long long i = 0; i < n; ++i) {
            y[i] = a * x[i] + y[i];
        }
    }

    return omp_get_wtime() - t0;
}

static double run_simd(double *x, double *y, long long n, int reps, double a)
{
    double t0 = omp_get_wtime();

    for (int r = 0; r < reps; ++r) {
        #pragma omp simd
        for (long long i = 0; i < n; ++i) {
            y[i] = a * x[i] + y[i];
        }
    }

    return omp_get_wtime() - t0;
}

static double run_parallel_simd(double *x, double *y, long long n, int reps, double a)
{
    double t0 = omp_get_wtime();

    for (int r = 0; r < reps; ++r) {
        #pragma omp parallel for simd default(none) shared(x, y, n, a) schedule(static)
        for (long long i = 0; i < n; ++i) {
            y[i] = a * x[i] + y[i];
        }
    }

    return omp_get_wtime() - t0;
}

/* ---------- main ---------- */

int main(int argc, char *argv[])
{
    const long long default_n = 50000000LL;
    const int default_reps = 5;

    long long n = parse_ll_or_default(argc, argv, 1, default_n);
    int reps = parse_int_or_default(argc, argv, 2, default_reps);

    printf("OpenMP SIMD introduction\n");
    printf("Kernel: y[i] = a*x[i] + y[i]\n");
    printf("N = %lld, reps = %d\n", n, reps);
    printf("Max threads available: %d\n\n", omp_get_max_threads());

    const double a = 1.000001;

    /*
     * Allocate vectors with a typical alignment for SIMD-friendly loads/stores.
     * 64 bytes is a common alignment choice on modern systems.
     */
    const size_t alignment = 64;

    double *x = (double *)aligned_alloc_best_effort(alignment, (size_t)n * sizeof(double));
    double *y0 = (double *)aligned_alloc_best_effort(alignment, (size_t)n * sizeof(double));
    double *y1 = (double *)aligned_alloc_best_effort(alignment, (size_t)n * sizeof(double));
    double *y2 = (double *)aligned_alloc_best_effort(alignment, (size_t)n * sizeof(double));

    /* Initialize x and a baseline y, then copy into each variant buffer. */
    init_vectors(x, y0, n);
    memcpy(y1, y0, (size_t)n * sizeof(double));
    memcpy(y2, y0, (size_t)n * sizeof(double));

    /* Serial baseline */
    double t_serial = run_serial(x, y0, n, reps, a);
    double c0 = checksum(y0, n);

    /* SIMD-only */
    double t_simd = run_simd(x, y1, n, reps, a);
    double c1 = checksum(y1, n);

    /* Parallel + SIMD */
    double t_par_simd = run_parallel_simd(x, y2, n, reps, a);
    double c2 = checksum(y2, n);

    printf("Timings:\n");
    printf("  serial:           %.6f s\n", t_serial);
    printf("  omp simd:         %.6f s\n", t_simd);
    printf("  parallel for simd %.6f s\n\n", t_par_simd);

    printf("Checksums:\n");
    printf("  serial:           %.6f\n", c0);
    printf("  omp simd:         %.6f\n", c1);
    printf("  parallel for simd %.6f\n\n", c2);

    /*
     * Validate that all variants produced identical results (within exact equality).
     * For this deterministic kernel and identical operation order within each variant,
     * equality is typically expected, but floating-point differences are possible
     * across compilers/flags. For a production test, prefer an epsilon comparison.
     */
    if (c0 == c1 && c0 == c2) {
        printf("Result check: PASS (checksums match exactly)\n\n");
    } else {
        printf("Result check: WARNING (checksums differ)\n");
        printf("  Differences may be caused by floating-point reassociation or compiler flags.\n\n");
    }

    printf("Interpretation:\n");
    printf("  - omp simd requests vectorization of the loop body within a single thread.\n");
    printf("  - parallel for simd combines thread-level parallelism with SIMD within each thread.\n");
    printf("  - SIMD effectiveness depends on contiguous access, alignment, and absence of loop-carried dependencies.\n");
    printf("  - Always inspect compiler vectorization reports when teaching or tuning SIMD code.\n");

    free(x);
    free(y0);
    free(y1);
    free(y2);

    return 0;
}
