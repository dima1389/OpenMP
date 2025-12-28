/*
 * File:        omp_barrier.c
 *
 * Purpose:
 *   Demonstrates explicit synchronization of threads using the OpenMP barrier construct.
 *
 *   This program shows:
 *     - independent work performed by each thread before a barrier
 *     - a global synchronization point (#pragma omp barrier)
 *     - coordinated execution of a subsequent phase after all threads arrive
 *
 *   The example intentionally introduces staggered execution (via thread-dependent
 *   delays) so that the effect of the barrier is clearly observable.
 *
 * Key concepts:
 *   - Explicit barriers in shared-memory parallel programs
 *   - Phase-based parallel execution
 *   - Relationship between implicit and explicit barriers
 *   - Correct placement of barriers to enforce ordering
 *
 * OpenMP features used:
 *   Directives:
 *     - parallel
 *     - barrier
 *     - single
 *
 *   Runtime library:
 *     - omp_get_thread_num()
 *     - omp_get_num_threads()
 *     - omp_get_max_threads()
 *     - omp_get_wtime()
 *
 * Compilation (GCC / Clang):
 *   gcc -O2 -Wall -Wextra -Wpedantic -g -fopenmp \
 *       omp_barrier.c -o omp_barrier
 *
 * Execution:
 *   ./omp_barrier
 *
 * Notes:
 *   - Barriers serialize progress across phases and can reduce parallel efficiency
 *     if overused.
 *   - Many OpenMP constructs (e.g., parallel for) include implicit barriers.
 *   - Explicit barriers should be used only when a true phase dependency exists.
 *
 * References:
 *   - OpenMP API Specification (OpenMP ARB): barrier construct
 */

#include <stdio.h>
#include <omp.h>

/*
 * Perform synthetic work for approximately 'delay' seconds.
 * Uses busy-waiting via omp_get_wtime() for portability.
 */
static void busy_delay(double delay)
{
    double t0 = omp_get_wtime();
    while ((omp_get_wtime() - t0) < delay) {
        /* busy wait */
    }
}

int main(void)
{
    printf("OpenMP barrier synchronization example\n");
    printf("Max threads available: %d\n\n", omp_get_max_threads());

    double global_start = omp_get_wtime();

    #pragma omp parallel default(none) shared(global_start)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        /*
         * Phase 1: independent work.
         * Each thread waits for a different amount of time to create skew.
         */
        double local_delay = 0.1 * (double)(tid + 1);
        busy_delay(local_delay);

        printf("Thread %d/%d reached barrier after %.2f s\n",
               tid, nthreads, omp_get_wtime() - global_start);

        /*
         * Explicit barrier:
         * All threads must arrive here before any can proceed.
         */
        #pragma omp barrier

        /*
         * Phase 2: coordinated work after the barrier.
         * The barrier guarantees that all Phase 1 work is complete.
         */
        #pragma omp single
        {
            printf("\nAll threads have reached the barrier.\n");
            printf("Entering Phase 2.\n\n");
        }

        printf("Thread %d/%d executing Phase 2 at %.2f s\n",
               tid, nthreads, omp_get_wtime() - global_start);
    }

    printf("\nBack to serial execution.\n");

    return 0;
}
