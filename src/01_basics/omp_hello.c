/*
 * File:        omp_hello.c
 *
 * Purpose:
 *   Minimal, canonical "Hello, World" example for OpenMP.
 *
 *   This program introduces the fundamental execution model of OpenMP:
 *     - a single initial (master) thread
 *     - creation of a team of threads via a parallel region
 *     - identification of threads using thread IDs
 *
 *   It is intended as the *first* runnable OpenMP program in the project and
 *   serves as a baseline for all subsequent examples.
 *
 * Key concepts:
 *   - OpenMP parallel region (#pragma omp parallel)
 *   - Thread identification (omp_get_thread_num)
 *   - Team size query (omp_get_num_threads)
 *   - Interaction with environment variables (OMP_NUM_THREADS)
 *
 * OpenMP features used:
 *   Directives:
 *     - parallel
 *
 *   Runtime library:
 *     - omp_get_thread_num()
 *     - omp_get_num_threads()
 *     - omp_get_max_threads()
 *
 * Compilation (GCC / Clang):
 *   gcc -O2 -Wall -Wextra -Wpedantic -g -fopenmp \
 *       omp_hello.c -o omp_hello
 *
 * Execution:
 *   ./omp_hello
 *
 *   Optional environment variables:
 *     export OMP_NUM_THREADS=4
 *
 * Notes:
 *   - Output order is intentionally unspecified; thread scheduling is non-deterministic.
 *   - This program demonstrates *parallel execution*, not performance scaling.
 *
 * References:
 *   - OpenMP API Specification (OpenMP ARB): parallel construct
 */

#include <stdio.h>
#include <omp.h>

int main(void)
{
    /*
     * The program starts execution with a single thread (the initial thread).
     * omp_get_max_threads() reports the maximum number of threads that *may*
     * be used when entering a parallel region.
     */
    printf("OpenMP hello example\n");
    printf("Maximum available threads: %d\n\n", omp_get_max_threads());

    /*
     * Enter a parallel region.
     * The OpenMP runtime creates a team of threads, each executing the same block.
     */
    #pragma omp parallel default(none)
    {
        /*
         * Each thread has a unique integer identifier in the range:
         *   [0, omp_get_num_threads() - 1]
         */
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        /*
         * Standard output is shared; interleaving of lines is expected.
         * This is intentional and illustrates concurrent execution.
         */
        printf("Hello from thread %d of %d\n", tid, nthreads);
    }

    /*
     * After the parallel region, execution continues with a single thread again.
     */
    printf("\nBack to serial execution.\n");

    return 0;
}
