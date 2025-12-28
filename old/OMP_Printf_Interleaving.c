/*
 * OpenMP printf() interleaving demonstration + suppression using #pragma omp critical
 *
 * What this shows:
 *   1) UNSAFE printing: multiple threads emit a message in *several* printf() calls.
 *      Because stdout is a shared resource, the output can interleave at token-level, e.g.:
 *        "Phase Phase 1 1 thread thread 0 1"
 *
 *   2) SAFE printing: the same multi-printf message is wrapped in a critical section,
 *      so each thread's full message appears contiguously.
 *
 * Compile (GCC / MinGW-w64 / Linux):
 *   gcc -O2 -Wall -Wextra -fopenmp OMP_Printf_Interleaving.c -o OMP_Printf_Interleaving
 *
 * Run:
 *   OMP_Printf_Interleaving
 *
 * Optional:
 *   - Control threads:
 *       export OMP_NUM_THREADS=8        (Linux/macOS)
 *       set OMP_NUM_THREADS=8           (Windows CMD)
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/* Busy-wait delay to increase the likelihood of printf() output interleaving. */
static void spin_delay(double seconds)
{
    double t0 = omp_get_wtime();
    while ((omp_get_wtime() - t0) < seconds) {
        /* busy wait */
    }
}

static void print_phase_message_unsafe(int phase, int tid, int nthreads)
{
    /* Intentionally split into multiple printf() calls (no newline until the end). */
    printf("Phase ");
    spin_delay(0.0005);

    printf("%d ", phase);
    spin_delay(0.0005);

    printf("thread ");
    spin_delay(0.0005);

    printf("%d ", tid);
    spin_delay(0.0005);

    printf("of ");
    spin_delay(0.0005);

    printf("%d\n", nthreads);

    /* Force output to appear quickly so you can *see* the interleaving */
    fflush(stdout);
}

static void print_phase_message_safe(int phase, int tid, int nthreads)
{
    /* Same multi-printf approach, but serialized via critical. */
    #pragma omp critical
    {
        printf("Phase ");
        spin_delay(0.0005);

        printf("%d ", phase);
        spin_delay(0.0005);

        printf("thread ");
        spin_delay(0.0005);

        printf("%d ", tid);
        spin_delay(0.0005);

        printf("of ");
        spin_delay(0.0005);

        printf("%d\n", nthreads);

        fflush(stdout);
    }
}

int main(int argc, char **argv)
{
    (void)argc; (void)argv;

    printf("\nOpenMP reports %d processors\n", omp_get_num_procs());
    printf("Max threads (runtime limit): %d\n\n", omp_get_max_threads());

    printf("========== DEMO 1: UNSAFE printing (expect interleaving) ==========\n\n");

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        /* Synchronize threads so they reach printf together, making interleaving easier to see. */
        #pragma omp barrier

        /* Phase 1, intentionally unsafe */
        print_phase_message_unsafe(1, tid, nthreads);

        /* Barrier does NOT “fix” printing; it only synchronizes progress */
        #pragma omp barrier

        /* Phase 2, still intentionally unsafe */
        print_phase_message_unsafe(2, tid, nthreads);
    }

    printf("\n========== DEMO 2: SAFE printing with #pragma omp critical ==========\n\n");

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        #pragma omp barrier
        print_phase_message_safe(1, tid, nthreads);

        #pragma omp barrier
        print_phase_message_safe(2, tid, nthreads);
    }

    printf("\nNotes:\n");
    printf("  - Interleaving is nondeterministic; increase OMP_NUM_THREADS if needed.\n");
    printf("  - A barrier only ensures all threads reach a point; it does not serialize stdout.\n");
    printf("  - critical serializes the *whole* print sequence, preventing token-level mixing.\n");

    return 0;
}
