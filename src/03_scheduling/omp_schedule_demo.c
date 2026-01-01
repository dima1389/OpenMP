/*
 * File:        omp_schedule_demo.c
 *
 * Purpose:
 *   Demonstrates OpenMP loop scheduling strategies and their practical impact on:
 *     - load balance (distribution of work among threads),
 *     - runtime overhead (scheduling cost),
 *     - and overall execution time.
 *
 *   The program simulates a loop where each iteration has a variable cost.
 *   It then executes the same loop with multiple scheduling policies:
 *     - static
 *     - dynamic
 *     - guided
 *     - runtime (controlled via OMP_SCHEDULE)
 *
 * Key concepts:
 *   - Work-sharing: #pragma omp parallel for
 *   - schedule(kind[, chunk]) semantics and load balancing behavior
 *   - Interactions with environment variables: OMP_SCHEDULE
 *   - Timing with omp_get_wtime()
 *
 * OpenMP features used:
 *   Directives:
 *     - parallel for
 *     - reduction
 *
 *   Runtime library:
 *     - omp_get_max_threads()
 *     - omp_get_num_threads()
 *     - omp_get_thread_num()
 *     - omp_get_wtime()
 *     - omp_get_schedule() (optional introspection)
 *
 * Compilation (GCC/Clang):
 *   gcc -O2 -Wall -Wextra -Wpedantic -g -fopenmp omp_schedule_demo.c -o omp_schedule_demo
 *
 * Execution:
 *   ./omp_schedule_demo [N] [pattern]
 *
 *   Arguments:
 *     N        : Number of loop iterations (default: 50,000,000)
 *     pattern  : Workload pattern selector (default: 1)
 *               1 = heavy-at-end    (iterations get more expensive with i)
 *               2 = heavy-at-start  (iterations get cheaper with i)
 *               3 = periodic        (periodic expensive spikes)
 *
 * Examples:
 *   ./omp_schedule_demo 50000000 1
 *
 *   Runtime schedule controlled by environment:
 *     export OMP_SCHEDULE="dynamic,1024"
 *     ./omp_schedule_demo 50000000 3
 *
 * Notes:
 *   - Output order is deterministic (single-thread prints), but measured timing depends
 *     on OS scheduling, CPU frequency scaling, and system load.
 *   - The workload simulation is purely computational to avoid I/O effects.
 *
 * References:
 *   - OpenMP API Specification (OpenMP ARB): schedule clause, OMP_SCHEDULE
 *   - GCC libgomp / LLVM libomp documentation (runtime behavior)
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
        exit(1);
    }

    return v;
}

/* Parse a positive int from argv with validation. */
static int parse_int_or_default(int argc, char *argv[], int index, int default_value)
{
    if (argc <= index) {
        return default_value;
    }

    errno = 0;
    char *end = NULL;
    long v = strtol(argv[index], &end, 10);

    if (errno != 0 || end == argv[index] || *end != '\0' || v <= 0 || v > 1000000L) {
        fprintf(stderr, "Invalid integer value at argv[%d]: '%s'\n", index, argv[index]);
        exit(1);
    }

    return (int)v;
}

/*
 * workload_units(i, n, pattern) returns a small integer representing
 * how much work iteration i should do. Larger means more work.
 *
 * The intent is to create imbalance if scheduling is poorly chosen.
 */
static int workload_units(long long i, long long n, int pattern)
{
    if (pattern == 1) {
        /* heavy-at-end: cheap early iterations, expensive late iterations */
        double x = (double)i / (double)n;
        int u = 1 + (int)(200.0 * x * x); /* increases towards the end */
        return u;
    } else if (pattern == 2) {
        /* heavy-at-start: expensive early iterations, cheap late iterations */
        double x = (double)i / (double)n;
        int u = 1 + (int)(200.0 * (1.0 - x) * (1.0 - x));
        return u;
    } else {
        /* periodic spikes: mostly cheap, occasional expensive bursts */
        const long long period = 10000;
        const long long spike_width = 250;
        long long phase = i % period;
        if (phase < spike_width) {
            return 250; /* spike */
        }
        return 2; /* baseline */
    }
}

/*
 * burn_cpu(units) performs deterministic CPU work so that each iteration
 * takes time roughly proportional to 'units'. This avoids I/O and sleep calls.
 */
static double burn_cpu(int units)
{
    /* Volatile prevents aggressive optimization from removing the loop. */
    volatile double acc = 0.0;

    /*
     * Keep the loop bounded and deterministic. The constant factor here is tuned
     * only to create measurable differences; adjust if needed for your machine.
     */
    const int inner = 200;

    for (int u = 0; u < units; ++u) {
        for (int k = 0; k < inner; ++k) {
            acc += (double)u * 0.000001 + (double)k * 0.0000001;
        }
    }

    return (double)acc;
}

static const char *sched_name(omp_sched_t kind)
{
    switch (kind) {
        case omp_sched_static:  return "static";
        case omp_sched_dynamic: return "dynamic";
        case omp_sched_guided:  return "guided";
        case omp_sched_auto:    return "auto";
        default:                return "unknown";
    }
}

/*
 * run_schedule(kind, chunk, n, pattern) executes the same loop with a chosen schedule.
 * The returned value is the elapsed time.
 *
 * We use a reduction to accumulate a dummy value, preventing the compiler from
 * optimizing away the loop body.
 *
 * Chunk semantics:
 *   - For schedule(static|dynamic|guided, chunk), OpenMP requires chunk > 0.
 *   - For schedule(runtime), chunk is controlled by OMP_SCHEDULE, so we ignore it.
 */
static double run_schedule(omp_sched_t kind, int chunk, long long n, int pattern)
{
    /* OpenMP requires a positive chunk for schedule(kind,chunk). */
    if (chunk <= 0) {
        chunk = 1;
    }

    double sum = 0.0;
    double t0 = omp_get_wtime();

    /*
     * We use separate pragmas because OpenMP requires compile-time known schedule kinds
     * unless using schedule(runtime).
     */
    if (kind == omp_sched_static) {
        #pragma omp parallel for default(none) shared(n, pattern, chunk) reduction(+:sum) schedule(static, chunk)
        for (long long i = 0; i < n; ++i) {
            int units = workload_units(i, n, pattern);
            sum += burn_cpu(units);
        }
    } else if (kind == omp_sched_dynamic) {
        #pragma omp parallel for default(none) shared(n, pattern, chunk) reduction(+:sum) schedule(dynamic, chunk)
        for (long long i = 0; i < n; ++i) {
            int units = workload_units(i, n, pattern);
            sum += burn_cpu(units);
        }
    } else if (kind == omp_sched_guided) {
        #pragma omp parallel for default(none) shared(n, pattern, chunk) reduction(+:sum) schedule(guided, chunk)
        for (long long i = 0; i < n; ++i) {
            int units = workload_units(i, n, pattern);
            sum += burn_cpu(units);
        }
    } else {
        /* runtime schedule: controlled via OMP_SCHEDULE */
        #pragma omp parallel for default(none) shared(n, pattern) reduction(+:sum) schedule(runtime)
        for (long long i = 0; i < n; ++i) {
            int units = workload_units(i, n, pattern);
            sum += burn_cpu(units);
        }
    }

    double t1 = omp_get_wtime();
    double elapsed = t1 - t0;

    /*
     * Print sum to keep the reduction "observable" and discourage dead-code elimination.
     * We do not interpret the numeric value; it is only a computation sink.
     */
    if (sum == 0.123456789) {
        printf("Impossible value: %f\n", sum);
    }

    return elapsed;
}

int main(int argc, char *argv[])
{
    const long long default_n = 50000000LL;
    const int default_pattern = 1;

    long long n = parse_ll_or_default(argc, argv, 1, default_n);
    int pattern = parse_int_or_default(argc, argv, 2, default_pattern);

    if (pattern < 1 || pattern > 3) {
        fprintf(stderr, "Invalid pattern: %d (valid: 1..3)\n", pattern);
        return 1;
    }

    printf("OpenMP scheduling demonstration\n");
    printf("N = %lld iterations, pattern = %d\n", n, pattern);
    printf("Max threads available: %d\n", omp_get_max_threads());

    /* Report runtime schedule setting (if used) */
    omp_sched_t current_kind = omp_sched_static;
    int current_chunk = 0;
    omp_get_schedule(&current_kind, &current_chunk);
    printf("Current OpenMP runtime schedule: %s, chunk = %d\n\n",
           sched_name(current_kind), current_chunk);

    /*
     * Chunk size:
     * For didactic simplicity, we use chunk=1 in static/dynamic/guided runs.
     * A companion example can explore chunk size effects in more depth by varying
     * the chunk parameter passed to run_schedule().
     */
    double t_static  = run_schedule(omp_sched_static,  1, n, pattern);
    double t_dynamic = run_schedule(omp_sched_dynamic, 1, n, pattern);
    double t_guided  = run_schedule(omp_sched_guided,  1, n, pattern);
    double t_runtime = run_schedule(omp_sched_auto,    0, n, pattern); /* uses schedule(runtime) */

    printf("Timings (seconds):\n");
    printf("  schedule(static,1):   %.6f\n", t_static);
    printf("  schedule(dynamic,1):  %.6f\n", t_dynamic);
    printf("  schedule(guided,1):   %.6f\n", t_guided);
    printf("  schedule(runtime):    %.6f  (set OMP_SCHEDULE)\n", t_runtime);

    printf("\nInterpretation guidelines:\n");
    printf("  - static:  lowest overhead, but can load-imbalance for uneven iteration costs.\n");
    printf("  - dynamic: better balance, higher overhead due to runtime work assignment.\n");
    printf("  - guided:  starts with large chunks, decreases chunk size; often good balance.\n");
    printf("  - runtime: schedule chosen externally via OMP_SCHEDULE for experimentation.\n");

    printf("\nExamples:\n");
    printf("  export OMP_SCHEDULE=\"dynamic,1024\"; ./omp_schedule_demo %lld %d\n", n, pattern);
    printf("  export OMP_SCHEDULE=\"guided\";       ./omp_schedule_demo %lld %d\n", n, pattern);

    return 0;
}
