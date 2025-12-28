/*
 * File:        omp_printf_interleaving.c
 *
 * Purpose:
 *   Demonstrates output interleaving when multiple OpenMP threads write to
 *   standard output concurrently using printf().
 *
 *   This example illustrates:
 *     - why printf() output from parallel regions may appear interleaved
 *     - that standard I/O is shared among threads
 *     - how lack of synchronization affects observable program behavior
 *
 *   The program contrasts:
 *     1) unsynchronized printf calls (interleaved output)
 *     2) synchronized printf calls using a critical section (ordered lines)
 *
 * Key concepts:
 *   - Shared I/O streams in multithreaded programs
 *   - Interleaving vs atomicity of output operations
 *   - Synchronization for observable behavior (not correctness)
 *
 * OpenMP features used:
 *   Directives:
 *     - parallel
 *     - critical
 *     - barrier
 *
 *   Runtime library:
 *     - omp_get_thread_num()
 *     - omp_get_num_threads()
 *     - omp_get_max_threads()
 *
 * Compilation (GCC / Clang):
 *   gcc -O2 -Wall -Wextra -Wpedantic -g -fopenmp \
 *       omp_printf_interleaving.c -o omp_printf_interleaving
 *
 * Execution:
 *   ./omp_printf_interleaving
 *
 * Notes:
 *   - printf() is typically thread-safe, but not atomic with respect to entire lines.
 *   - Interleaving affects readability, not program correctness.
 *   - Synchronizing I/O can significantly reduce parallel performance and should
 *     generally be avoided in performance-critical code.
 *
 * References:
 *   - OpenMP API Specification (OpenMP ARB): critical construct
 *   - ISO C standard: stdio thread safety (implementation-defined details)
 */

#include <stdio.h>
#include <omp.h>

int main(void)
{
    printf("OpenMP printf interleaving demonstration\n");
    printf("Max threads available: %d\n\n", omp_get_max_threads());

    /*
     * Part 1: Unsynchronized output.
     * Lines printed here may be interleaved or mixed across threads.
     */
    printf("=== Unsynchronized printf() ===\n");

    #pragma omp parallel default(none)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        for (int i = 0; i < 3; ++i) {
            printf("Thread %d/%d says hello (%d)\n", tid, nthreads, i);
        }
    }

    /*
     * Ensure all output from the first section completes before continuing.
     */
    #pragma omp barrier

    printf("\n=== Synchronized printf() using critical ===\n");

    /*
     * Part 2: Synchronized output using a critical section.
     * Each printf() call is serialized, preventing interleaving.
     */
    #pragma omp parallel default(none)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        for (int i = 0; i < 3; ++i) {
            #pragma omp critical
            {
                printf("Thread %d/%d says hello (%d)\n", tid, nthreads, i);
            }
        }
    }

    printf("\nBack to serial execution.\n");

    return 0;
}
