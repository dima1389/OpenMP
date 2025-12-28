/*
 * File:        omp_schedule_profile.c
 *
 * Purpose:
 *   Makes OpenMP scheduling behavior *visible* by collecting per-thread statistics:
 *     - number of iterations executed per thread
 *     - total "work units" processed per thread (proxy for time/effort)
 *
 *   The program runs the same non-uniform workload loop under different schedules:
 *     - schedule(static,  chunk)
 *     - schedule(dynamic, chunk)
 *     - schedule(guided,  chunk)
 *     - schedule(runtime) (controlled by OMP_SCHEDULE)
 *
 *   It then prints a compact "work distribution profile" to illustrate:
 *     - load imbalance patterns
 *     - how scheduling strategies compensate (or not)
 *     - how chunk size affects fairness and overhead
 *
 * Key concepts:
 *   - Work-sharing schedules and chunking
 *   - Load balance vs scheduling overhead
 *   - Per-thread accounting in shared-memory programs
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
 *     - omp_get_schedule()
 *     - omp_get_wtime()
 *
 * Compilation (GCC/Clang):
 *   gcc -O2 -Wall -Wextra -Wpedantic -g -fopenmp \
 *       omp_schedule_profile.c -o omp_schedule_profile
 *
 * Execution:
 *   ./omp_schedule_profile [N] [pattern] [chunk]
 *
 *   Arguments:
 *     N        : number of iterations (default: 20,000,000)
 *     pattern  : workload pattern
 *                1 = heavy-at-end
 *                2 = heavy-at-start
 *                3 = periodic spikes
 *     chunk    : chunk size for static/dynamic/guided (default: 1)
 *
 * Examples:
 *   ./omp_schedule_profile 20000000 1 1
 *   ./omp_schedule_profile 20000000 1 1024
 *
 *   export OMP_SCHEDULE="dynamic,4096"
 *   ./omp_schedule_profile 20000000 3 64
 *
 * Notes:
 *   - This example prints per-thread statistics only; it does not attempt to enforce
 *     output ordering beyond printing from a single thread after the parallel region.
 *   - Work units are synthetic; they approximate relative iteration cost.
 *
 * References:
 *   - OpenMP API Specification (OpenMP ARB): schedule clause, OMP_SCHEDULE
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

/* ---------- per-thread statistics ---------- */

typedef struct {
    long long iters;     /* number of loop iterations executed by the thread */
    long long units;     /* total work units processed by the thread */
} thread_stats_t;

static void stats_zero(thread_stats_t *stats, int nthreads)
{
    for (int t = 0; t < nthreads; ++t) {
        stats[t].iters = 0;
        stats[t].units = 0;
    }
}

static void stats_print(const char *label, const thread_stats_t *stats, int nthreads)
{
    long long total_iters = 0;
    long long total_units = 0;

    long long min_iters = (nthreads > 0) ? stats[0].iters : 0;
    long long max_iters = (nthreads > 0) ? stats[0].iters : 0;

    long long min_units = (nthreads > 0) ? stats[0].units : 0;
    long long max_units = (nthreads > 0) ? stats[0].units : 0;

    printf("%s\n", label);
    printf("Thread | Iterations | WorkUnits\n");
    printf("-------+------------+----------\n");

    for (int t = 0; t < nthreads; ++t) {
        printf("%6d | %10lld | %8lld\n", t, stats[t].iters, stats[t].units);

        total_iters += stats[t].iters;
        total_units += stats[t].units;

        if (stats[t].iters < min_iters) min_iters = stats[t].iters;
        if (stats[t].iters > max_iters) max_iters = stats[t].iters;

        if (stats[t].units < min_units) min_units = stats[t].units;
        if (stats[t].units > max_units) max_units = stats[t].units;
    }

    printf("-------+------------+----------\n");
    printf("Total  | %10lld | %8lld\n", total_iters, total_units);

    printf("\nBalance metrics:\n");
    printf("  Iterations: min=%lld, max=%lld, ratio(max/min)=%.3f\n",
           min_iters, max_iters,
           (min_iters > 0) ? (double)max_iters / (double)min_iters : 0.0);

    printf("  WorkUnits:  min=%lld, max=%lld, ratio(max/min)=%.3f\n\n",
           min_units, max_units,
           (min_units > 0) ? (double)max_units / (double)min_units : 0.0);
}

/* ---------- run loop under a chosen schedule ---------- */

static double run_profiled_loop(omp_sched_t sched_kind,
                                long long n, int pattern, int chunk,
                                thread_stats_t *stats, int nthreads)
{
    stats_zero(stats, nthreads);

    double t0 = omp_get_wtime();

    if (sched_kind == omp_sched_static) {
        #pragma omp parallel default(none) shared(n, pattern, chunk, stats, nthreads)
        {
            int tid = omp_get_thread_num();

            #pragma omp for schedule(static, chunk)
            for (long long i = 0; i < n; ++i) {
                int u = workload_units(i, n, pattern);
                stats[tid].iters += 1;
                stats[tid].units += (long long)u;
            }
        }
    } else if (sched_kind == omp_sched_dynamic) {
        #pragma omp parallel default(none) shared(n, pattern, chunk, stats, nthreads)
        {
            int tid = omp_get_thread_num();

            #pragma omp for schedule(dynamic, chunk)
            for (long long i = 0; i < n; ++i) {
                int u = workload_units(i, n, pattern);
                stats[tid].iters += 1;
                stats[tid].units += (long long)u;
            }
        }
    } else if (sched_kind == omp_sched_guided) {
        #pragma omp parallel default(none) shared(n, pattern, chunk, stats, nthreads)
        {
            int tid = omp_get_thread_num();

            #pragma omp for schedule(guided, chunk)
            for (long long i = 0; i < n; ++i) {
                int u = workload_units(i, n, pattern);
                stats[tid].iters += 1;
                stats[tid].units += (long long)u;
            }
        }
    } else {
        #pragma omp parallel default(none) shared(n, pattern, stats, nthreads)
        {
            int tid = omp_get_thread_num();

            #pragma omp for schedule(runtime)
            for (long long i = 0; i < n; ++i) {
                int u = workload_units(i, n, pattern);
                stats[tid].iters += 1;
                stats[tid].units += (long long)u;
            }
        }
    }

    double t1 = omp_get_wtime();
    return t1 - t0;
}

static const char *sched_name(omp_sched_t k)
{
    switch (k) {
        case omp_sched_static:  return "static";
        case omp_sched_dynamic: return "dynamic";
        case omp_sched_guided:  return "guided";
        case omp_sched_auto:    return "auto";
        default:                return "unknown";
    }
}

/* ---------- main ---------- */

int main(int argc, char *argv[])
{
    const long long default_n = 20000000LL;
    const int default_pattern = 1;
    const int default_chunk = 1;

    long long n = parse_ll_or_default(argc, argv, 1, default_n);
    int pattern = parse_int_or_default(argc, argv, 2, default_pattern);
    int chunk = parse_int_or_default(argc, argv, 3, default_chunk);

    if (pattern < 1 || pattern > 3) {
        fprintf(stderr, "Invalid pattern: %d (valid: 1..3)\n", pattern);
        return 1;
    }

    printf("OpenMP scheduling profiler (work distribution visibility)\n");
    printf("N = %lld, pattern = %d, chunk = %d\n", n, pattern, chunk);
    printf("Max threads available: %d\n", omp_get_max_threads());

    omp_sched_t rk;
    int rc;
    omp_get_schedule(&rk, &rc);
    printf("Runtime schedule (omp_get_schedule): kind=%s, chunk=%d\n\n", sched_name(rk), rc);

    /*
     * We allocate per-thread stats arrays sized by omp_get_max_threads().
     * The actual number of threads used can be lower (e.g., due to OMP_NUM_THREADS),
     * but the indexing remains safe.
     */
    int nthreads = omp_get_max_threads();
    thread_stats_t *stats = (thread_stats_t *)calloc((size_t)nthreads, sizeof(thread_stats_t));
    if (stats == NULL) {
        fprintf(stderr, "Allocation failure for per-thread stats.\n");
        return 1;
    }

    /* ---- run and print profiles ---- */

    double t_static = run_profiled_loop(omp_sched_static, n, pattern, chunk, stats, nthreads);
    stats_print("Schedule: static", stats, nthreads);
    printf("Elapsed time (static): %.6f s\n\n", t_static);

    double t_dynamic = run_profiled_loop(omp_sched_dynamic, n, pattern, chunk, stats, nthreads);
    stats_print("Schedule: dynamic", stats, nthreads);
    printf("Elapsed time (dynamic): %.6f s\n\n", t_dynamic);

    double t_guided = run_profiled_loop(omp_sched_guided, n, pattern, chunk, stats, nthreads);
    stats_print("Schedule: guided", stats, nthreads);
    printf("Elapsed time (guided): %.6f s\n\n", t_guided);

    double t_runtime = run_profiled_loop(omp_sched_auto, n, pattern, chunk, stats, nthreads);
    stats_print("Schedule: runtime (set via OMP_SCHEDULE)", stats, nthreads);
    printf("Elapsed time (runtime): %.6f s\n\n", t_runtime);

    printf("Guidance:\n");
    printf("  - Compare WorkUnits distribution across schedules to see load balance.\n");
    printf("  - Compare elapsed times to see overhead vs balance trade-offs.\n");
    printf("  - Try different OMP_SCHEDULE values, e.g.:\n");
    printf("      export OMP_SCHEDULE=\"dynamic,1024\"\n");
    printf("      export OMP_SCHEDULE=\"guided,64\"\n");

    free(stats);
    return 0;
}
