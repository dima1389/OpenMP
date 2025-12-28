/*
 * OpenMP barrier demonstration.
 *
 * Toolchain and runtime DLL consistency notice (Windows / MSYS2 MinGW-w64):
 *   If you have multiple MSYS2 environments installed (e.g., mingw64 vs ucrt64),
 *   ensure the intended toolchain's bin directory comes first in PATH. Otherwise,
 *   the program (or compiler helper executables) may load incompatible DLLs at
 *   runtime (a common symptom is an entry-point error involving zlib1.dll).
 *
 * Recommended environment alignment:
 *   set "PATH=C:\msys64\mingw64\bin;%PATH%"
 *
 * Compilation:
 *   gcc -fopenmp -Wall OMP_Barrier.c -o OMP_Barrier
 *
 * Execution:
 *   OMP_Barrier
 */

#include <stdio.h>  /* printf, fflush */
#include <omp.h>    /* OpenMP runtime API */

/*
 * This example demonstrates what an OpenMP barrier guarantees:
 *   - Synchronization: all threads in the team must reach the barrier
 *     before any thread is allowed to proceed beyond it.
 *   - Ordering (program-order around the barrier): code after the barrier
 *     cannot execute on any thread until every thread has completed the
 *     code before the barrier.
 *
 * Important practical note:
 *   Console output is a shared I/O side-effect and may be buffered. To make
 *   the observed output reliably reflect the barrier ordering, printing is
 *   serialized using an OpenMP 'critical' section.
 */
int main(void)
{
    printf("OpenMP reports %d processors\n", omp_get_num_procs());
    printf("Max threads (runtime limit): %d\n\n", omp_get_max_threads());

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        /* Phase 1: executed by each thread before the barrier.
         *
         * #pragma omp critical ensures thread-safe output:
         * - Prevents interleaved printf() calls (e.g., "Phase Phase 1 1 thread thread 0 1")
         * - fflush(stdout) forces immediate output
         */
        #pragma omp critical
        {
            printf("Phase 1 (before barrier): thread %d of %d\n", tid, nthreads);
            fflush(stdout);
        }

        /*
         * BARRIER:
         *   No thread can execute Phase 2 until ALL threads have printed Phase 1
         *   (i.e., reached this point in the program).
         */
        #pragma omp barrier

        /* Phase 2: executed by each thread after the barrier. */
        #pragma omp critical
        {
            printf("Phase 2 (after barrier):  thread %d of %d\n", tid, nthreads);
            fflush(stdout);
        }

        /*
         * Even with the barrier, the relative ordering *within* each phase
         * (which thread prints first) remains nondeterministic and depends on
         * the runtime scheduler and OS timing.
         */
    }

    /*
     * End of parallel region:
     *   All threads in the team join (implicit barrier).
     *   Execution continues with the encountering thread (here: the initial thread).
     */
    return 0;
}
