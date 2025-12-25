/*
 * Standard C header for input/output operations.
 * Provides printf().
 */
#include <stdio.h>

/*
 * Standard C library header.
 * Provides strtol(), which converts a string to an integer.
 */
#include <stdlib.h>

/*
 * OpenMP header file.
 * Enables OpenMP pragmas and runtime library functions such as:
 *   - omp_get_wtime()
 *   - omp_get_max_threads()
 */
#include <omp.h>

/*
 * PURPOSE OF THIS PROGRAM
 *
 * This program demonstrates DIFFERENT OpenMP LOOP SCHEDULING STRATEGIES:
 *
 *   1) static
 *   2) dynamic
 *   3) guided
 *   4) runtime
 *
 * All four versions compute the same mathematical result:
 *
 *     sum = 1 + 2 + 3 + ... + N
 *
 * The difference is HOW loop iterations are distributed among threads.
 *
 * Recommended environment alignment:
 *   set "PATH=C:\msys64\mingw64\bin;%PATH%"
 *
 * Compilation:
 *   gcc -fopenmp -Wall OMP_Schedule_Demo.c -o OMP_Schedule_Demo
 *
 * Example execution:
 *   OMP_Schedule_Demo 100000000
 *
 * Optional (only affects schedule(runtime)):
 *   export OMP_SCHEDULE="dynamic,4"
 */

/*
 * Program entry point.
 */
int main(int argc, char *argv[])
{
    /*
     * Validate command-line arguments.
     *
     * This program expects EXACTLY ONE argument:
     *   argv[1] = N (upper limit of the summation)
     */
    if (argc != 2) {
        printf("Usage: %s <N>\n", argv[0]);
        return 1;
    }

    /*
     * Convert command-line argument from string to integer.
     *
     * strtol():
     *   - argv[1] : input string
     *   - NULL    : we do not need the end pointer
     *   - 10      : base-10 conversion
     *
     * "n" represents the total number of loop iterations.
     */
    long n = strtol(argv[1], NULL, 10);

    /*
     * "sum" will store the final summation result.
     * It is reused across multiple experiments.
     */
    double sum;

    /*
     * Timing variables for performance measurement.
     *
     * omp_get_wtime() returns elapsed wall-clock time in seconds.
     */
    double start, end;

    /*
     * Print the maximum number of threads OpenMP is allowed to use.
     *
     * This value depends on:
     *   - System hardware
     *   - Environment variables (OMP_NUM_THREADS)
     *   - OpenMP runtime defaults
     */
    printf("Number of threads: %d\n\n", omp_get_max_threads());

    /* ============================================================
     * STATIC SCHEDULE
     * ============================================================
     *
     * schedule(static)
     *
     * Behavior:
     *   - Loop iterations are divided into contiguous blocks
     *   - Blocks are assigned to threads BEFORE execution starts
     *
     * Example (N = 16, 4 threads):
     *   Thread 0: iterations 1–4
     *   Thread 1: iterations 5–8
     *   Thread 2: iterations 9–12
     *   Thread 3: iterations 13–16
     *
     * Characteristics:
     *   - Very low scheduling overhead
     *   - Best for uniform workloads
     *   - Poor load balance if iterations take unequal time
     */
    sum = 0.0;
    start = omp_get_wtime();

#pragma omp parallel for reduction(+:sum) schedule(static)
    for (long i = 1; i <= n; ++i) {
        sum += i;
    }

    end = omp_get_wtime();
    printf("STATIC   schedule: sum = %.0f, time = %f s\n", sum, end - start);

    /* ============================================================
     * DYNAMIC SCHEDULE
     * ============================================================
     *
     * schedule(dynamic, 1000)
     *
     * Behavior:
     *   - Loop iterations are split into chunks of size 1000
     *   - Threads request a new chunk ONLY after finishing their current one
     *
     * Characteristics:
     *   - Good load balancing for irregular workloads
     *   - Higher scheduling overhead than static
     *   - Threads may execute non-contiguous iteration ranges
     *
     * Example:
     *   Thread 0: 1–1000, 4001–5000, ...
     *   Thread 1: 1001–2000, 5001–6000, ...
     */
    sum = 0.0;
    start = omp_get_wtime();

#pragma omp parallel for reduction(+:sum) schedule(dynamic, 1000)
    for (long i = 1; i <= n; ++i) {
        sum += i;
    }

    end = omp_get_wtime();
    printf("DYNAMIC  schedule: sum = %.0f, time = %f s\n", sum, end - start);

    /* ============================================================
     * GUIDED SCHEDULE
     * ============================================================
     *
     * schedule(guided, 1000)
     *
     * Behavior:
     *   - Similar to dynamic scheduling
     *   - Chunk size starts LARGE and gradually decreases
     *   - Chunk size never becomes smaller than the given minimum (1000)
     *
     * Purpose:
     *   - Reduce scheduling overhead at the beginning
     *   - Improve load balance toward the end
     *
     * Commonly used for:
     *   - Large iteration spaces
     *   - Workloads with unpredictable execution time
     */
    sum = 0.0;
    start = omp_get_wtime();

#pragma omp parallel for reduction(+:sum) schedule(guided, 1000)
    for (long i = 1; i <= n; ++i) {
        sum += i;
    }

    end = omp_get_wtime();
    printf("GUIDED   schedule: sum = %.0f, time = %f s\n", sum, end - start);

    /* ============================================================
     * RUNTIME SCHEDULE
     * ============================================================
     *
     * schedule(runtime)
     *
     * Behavior:
     *   - The scheduling policy is NOT fixed at compile time
     *   - The OpenMP runtime reads it from the OMP_SCHEDULE
     *     environment variable at execution time
     *
     * Example:
     *   export OMP_SCHEDULE="dynamic,4"
     *
     * Advantages:
     *   - Same binary can be tested with different schedules
     *   - No recompilation required
     *
     * If OMP_SCHEDULE is not set:
     *   - A default (usually static) is used
     */
    sum = 0.0;
    start = omp_get_wtime();

#pragma omp parallel for reduction(+:sum) schedule(runtime)
    for (long i = 1; i <= n; ++i) {
        sum += i;
    }

    end = omp_get_wtime();
    printf("RUNTIME  schedule: sum = %.0f, time = %f s\n", sum, end - start);

    /*
     * Successful program termination.
     */
    return 0;
}
