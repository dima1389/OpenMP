/*
 * File:        omp_schedule_demo_chunks.c
 *
 * Purpose:
 *   Demonstrates the impact of OpenMP scheduling *and chunk size* on load balance,
 *   scheduling overhead, and total execution time.
 *
 *   This example extends omp_schedule_demo.c by making the chunk size an explicit,
 *   user-controlled parameter and by comparing:
 *
 *     - schedule(static,  chunk)
 *     - schedule(dynamic, chunk)
 *     - schedule(guided,  chunk)
 *     - schedule(runtime)  (controlled via OMP_SCHEDULE)
 *
 *   The workload per iteration is intentionally non-uniform to expose differences
 *   between scheduling strategies.
 *
 * Key concepts:
 *   - schedule(kind, chunk) semantics
 *   - Load imbalance vs scheduling overhead
 *   - Chunk size sensitivity
 *   - Runtime-controlled scheduling (OMP_SCHEDULE)
 *
 * OpenMP features used:
 *   Directives:
 *     - parallel for
 *     - reduction
 *
 *   Runtime library:
 *     - omp_get_max_threads()
 *     - omp_get_schedule()
 *     - omp_get_wtime()
 *
 * Compilation (GCC/Clang):
 *   gcc -O2 -Wall -Wextra -Wpedantic -g -fopenmp \
 *       omp_schedule_demo_chunks.c -o omp_schedule_demo_chunks
 *
 * Execution:
 *   ./omp_schedule_demo_chunks [N] [pattern] [chunk]
 *
 *   Arguments:
 *     N        : number of loop iterations (default: 50,000,000)
 *     pattern  : workload pattern
 *                1 = heavy-at-end
 *                2 = heavy-at-start
 *                3 = periodic spikes
 *     chunk    : chunk size for static/dynamic/guided (default: 1)
 *
 * Examples:
 *   ./omp_schedule_demo_chunks 50000000 1 1
 *   ./omp_schedule_demo_chunks 50000000 1 1024
 *
 *   export OMP_SCHEDULE="dynamic,4096"
 *   ./omp_schedule_demo_chunks 50000000 3 64
 *
 * Notes:
 *   - Output is deterministic; timing is system-dependent.
 *   - Chunk size has no effect for schedule(runtime) unless specified via OMP_SCHEDULE.
 *
 * References:
 *   - OpenMP API Specification (OpenMP ARB): schedule clause, OMP_SCHEDULE
 *   - GCC libgomp / LLVM libomp documentation
 */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <omp.h>

/* ---------- argument parsing helpers ---------- */

static long long parse_ll_or_default(int argc, char *argv[],
                                     int index, long long def)
{
    if (argc <= index) {
        return def;
    }

    errno = 0;
    char *end = NULL;
    long long v = strtoll(argv[index], &end, 10);

    if (errno != 0 || end == argv[index] || *end != '\0' || v <= 0) {
        fprintf(stderr, "Invalid numeric value at argv[%d]: '%s'\n",
                index, argv[index]);
        exit(1);
    }

    return v;
}

static int parse_int_or_default(int argc, char *argv[],
                                int index, int def)
{
    if (argc <= index) {
        return def;
    }

    errno = 0;
    char *end = NULL;
    long v = strtol(argv[index], &end, 10);

    if (errno != 0 || end == argv[index] || *end != '\0' || v <= 0) {
        fprintf(stderr, "Invalid integer value at argv[%d]: '%s'\n",
                index, argv[index]);
        exit(1);
    }

    return (int)v;
}

/* ---------- workload model ---------- */

static int workload_units(long long i, long long n, int pattern)
{
    if (pattern == 1) {
        /* heavy-at-end */
        double x = (double)i / (double)n;
        return 1 + (int)(200.0 * x * x);
    } else if (pattern == 2) {
        /* heavy-at-start */
        double x = (double)i / (double)n;
        return 1 + (int)(200.0 * (1.0 - x) * (1.0 - x));
    } else {
        /* periodic spikes */
        const long long period = 10000;
        const long long spike = 250;
        return ((i % period) < spike) ? 250 : 2;
    }
}

/* deterministic CPU work */
static double burn_cpu(int units)
{
    volatile double acc = 0.0;
    const int inner = 200;

    for (int u = 0; u < units; ++u) {
        for (int k = 0; k < inner; ++k) {
            acc += (double)u * 1e-6 + (double)k * 1e-7;
        }
    }

    return acc;
}

/* ---------- scheduling runner ---------- */

static double run_loop(omp_sched_t kind,
                       long long n, int pattern, int chunk)
{
    double sum = 0.0;
    double t0 = omp_get_wtime();

    if (kind == omp_sched_static) {
        #pragma omp parallel for default(none) shared(n, pattern, chunk) \
                reduction(+:sum) schedule(static, chunk)
        for (long long i = 0; i < n; ++i) {
            sum += burn_cpu(workload_units(i, n, pattern));
        }
    } else if (kind == omp_sched_dynamic) {
        #pragma omp parallel for default(none) shared(n, pattern, chunk) \
                reduction(+:sum) schedule(dynamic, chunk)
        for (long long i = 0; i < n; ++i) {
            sum += burn_cpu(workload_units(i, n, pattern));
        }
    } else if (kind == omp_sched_guided) {
        #pragma omp parallel for default(none) shared(n, pattern, chunk) \
                reduction(+:sum) schedule(guided, chunk)
        for (long long i = 0; i < n; ++i) {
            sum += burn_cpu(workload_units(i, n, pattern));
        }
    } else {
        /* runtime */
        #pragma omp parallel for default(none) shared(n, pattern) \
                reduction(+:sum) schedule(runtime)
        for (long long i = 0; i < n; ++i) {
            sum += burn_cpu(workload_units(i, n, pattern));
        }
    }

    double t1 = omp_get_wtime();

    /* prevent dead-code elimination */
    if (sum == 123456.0) {
        printf("Impossible sum: %f\n", sum);
    }

    return t1 - t0;
}

/* ---------- main ---------- */

int main(int argc, char *argv[])
{
    const long long default_n = 50000000LL;
    const int default_pattern = 1;
    const int default_chunk = 1;

    long long n = parse_ll_or_default(argc, argv, 1, default_n);
    int pattern = parse_int_or_default(argc, argv, 2, default_pattern);
    int chunk = parse_int_or_default(argc, argv, 3, default_chunk);

    if (pattern < 1 || pattern > 3) {
        fprintf(stderr, "Invalid pattern: %d (valid: 1..3)\n", pattern);
        return 1;
    }

    printf("OpenMP scheduling demo (chunk size sensitivity)\n");
    printf("N = %lld, pattern = %d, chunk = %d\n", n, pattern, chunk);
    printf("Max threads available: %d\n", omp_get_max_threads());

    omp_sched_t k;
    int c;
    omp_get_schedule(&k, &c);
    printf("Runtime schedule: kind=%d, chunk=%d\n\n", (int)k, c);

    double t_static  = run_loop(omp_sched_static,  n, pattern, chunk);
    double t_dynamic = run_loop(omp_sched_dynamic, n, pattern, chunk);
    double t_guided  = run_loop(omp_sched_guided,  n, pattern, chunk);
    double t_runtime = run_loop(omp_sched_auto,    n, pattern, chunk);

    printf("Timings (seconds):\n");
    printf("  static (%d):   %.6f\n", chunk, t_static);
    printf("  dynamic(%d):   %.6f\n", chunk, t_dynamic);
    printf("  guided (%d):   %.6f\n", chunk, t_guided);
    printf("  runtime:       %.6f  (OMP_SCHEDULE)\n", t_runtime);

    printf("\nInterpretation:\n");
    printf("  - Smaller chunks improve load balance but increase scheduling overhead.\n");
    printf("  - Larger chunks reduce overhead but risk load imbalance.\n");
    printf("  - dynamic/guided schedules benefit most from careful chunk tuning.\n");
    printf("  - runtime allows experimentation without recompilation.\n");

    return 0;
}
