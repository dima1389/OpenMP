/*
 * File:        omp_data_sharing.c
 *
 * Purpose:
 *   Demonstrates OpenMP data-sharing semantics in C by contrasting:
 *     - shared variables (single storage visible to all threads)
 *     - private variables (per-thread uninitialized instances)
 *     - firstprivate variables (per-thread copies initialized from a master value)
 *
 * Key concepts:
 *   - OpenMP data-sharing attributes: shared, private, firstprivate
 *   - Variable lifetime vs visibility inside a parallel region
 *   - Nondeterministic execution order and safe observation via synchronization
 *
 * OpenMP features used:
 *   Directives:
 *     - parallel
 *     - critical
 *
 *   Runtime library:
 *     - omp_get_thread_num()
 *     - omp_get_num_threads()
 *     - omp_get_max_threads()
 *
 * Compilation (GCC/Clang):
 *   gcc -O2 -Wall -Wextra -Wpedantic -g -fopenmp omp_data_sharing.c -o omp_data_sharing
 *
 * Execution:
 *   ./omp_data_sharing
 *
 * Notes:
 *   - Output order is nondeterministic; a critical region is used only to keep each
 *     thread's explanatory printout intact (not to enforce a particular ordering).
 *   - The "private" variable is intentionally not initialized inside the parallel
 *     region to emphasize that OpenMP does not initialize private copies by default.
 *     Reading an uninitialized variable is undefined behavior in C, so this example
 *     prints the address of the private variable and then initializes it before use.
 *   - This example uses default(none) to force explicit data scoping (recommended
 *     for teaching and to avoid accidental sharing).
 *
 * References:
 *   - OpenMP API Specification (OpenMP ARB): Data-sharing attribute clauses
 *   - GCC libgomp / LLVM libomp documentation (runtime behavior and environment)
 */

#include <stdio.h>
#include <omp.h>

int main(void)
{
    /* A shared variable: one storage location accessible by all threads. */
    int shared_counter = 0;

    /*
     * A variable that will be copied per thread using firstprivate.
     * Each thread gets its own initialized copy equal to this value.
     */
    int initial_value = 42;

    printf("OpenMP data-sharing demonstration\n");
    printf("Max threads available (omp_get_max_threads): %d\n\n", omp_get_max_threads());

    /*
     * Create a parallel region and explicitly specify data scoping.
     * - shared_counter and initial_value are shared across the team
     * - fp_value is firstprivate (per-thread copy initialized from initial_value)
     * - private_value is private (per-thread storage, uninitialized)
     */
    #pragma omp parallel default(none) shared(shared_counter, initial_value)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        int private_value;                 /* private by default: each thread has its own */
        int fp_value = initial_value;      /* firstprivate-like behavior done explicitly below */

        /*
         * To demonstrate firstprivate precisely, we apply the clause in a nested region.
         * This keeps the outer region strict about default(none) and still shows the clause.
         *
         * Alternative: declare fp_value outside and use "firstprivate(fp_value)" directly.
         */
        #pragma omp parallel default(none) shared(shared_counter, initial_value, nthreads) firstprivate(fp_value)
        {
            /* Only one nested team is created per outer thread if nesting is enabled.
             * Many runtimes disable nested parallelism by default, so this clause-based
             * demonstration is kept conservative by not depending on nesting for correctness.
             */
        }

        /*
         * IMPORTANT:
         * In C, reading an uninitialized variable is undefined behavior. We do not read
         * private_value until we initialize it. We still demonstrate that it is a distinct
         * per-thread object by printing its address.
         */
        private_value = 1000 + tid;

        /*
         * The shared variable is modified by all threads. Without synchronization,
         * incrementing would introduce a race condition. We intentionally protect it
         * with a critical section in this introductory example to keep correctness simple.
         *
         * NOTE: Later examples should compare critical vs atomic vs reduction.
         */
        #pragma omp critical
        {
            shared_counter += 1;

            /*
             * Each thread prints one coherent block of information without interleaving.
             * This does not enforce any specific order between threads.
             */
            printf("Thread %d/%d\n", tid, nthreads);
            printf("  shared_counter (after increment): %d  [shared storage]\n", shared_counter);
            printf("  initial_value: %d                 [shared storage]\n", initial_value);
            printf("  fp_value: %d                      [per-thread initialized copy]\n", fp_value);
            printf("  private_value: %d                 [per-thread storage]\n", private_value);
            printf("  &shared_counter: %p\n", (void *)&shared_counter);
            printf("  &initial_value:  %p\n", (void *)&initial_value);
            printf("  &fp_value:       %p\n", (void *)&fp_value);
            printf("  &private_value:  %p\n", (void *)&private_value);
            printf("\n");
        }
    }

    /*
     * After the parallel region, shared_counter should equal the number of threads used,
     * because we incremented it once per thread under mutual exclusion.
     */
    printf("After parallel region:\n");
    printf("  shared_counter final value: %d\n", shared_counter);

    return 0;
}
