/*
 * OpenMP "Hello" example (thread team creation + basic runtime queries).
 *
 * Windows note (MSYS2):
 *   If you have multiple MSYS2 toolchains installed (e.g., mingw64 vs ucrt64),
 *   ensure the intended toolchain's bin directory comes first in PATH.
 *   This avoids runtime DLL mismatches where the executable loads DLLs from a
 *   different toolchain than the one used at link time.
 *
 * Recommended environment alignment:
 *   set "PATH=C:\msys64\mingw64\bin;%PATH%"
 *
 * Compilation:
 *   gcc -fopenmp -Wall OMP_Hello.c -o OMP_Hello
 *
 * Execution:
 *   OMP_Hello
 *
 * Example output (depends on thread scheduling by the OS):
 *   OpenMP reports 8 processors
 *   Max threads (runtime limit): 8
 *   Hello from thread 2 of 8
 *   Hello from thread 0 of 8
 *   Hello from thread 6 of 8
 *   Hello from thread 7 of 8
 *   Hello from thread 4 of 8
 *   Hello from thread 3 of 8
 *   Hello from thread 1 of 8
 *   Hello from thread 5 of 8
 */

#include <stdio.h>  /* printf */
#include <omp.h>    /* OpenMP runtime API */

/*
 * This program demonstrates:
 *   - Creating a thread team with '#pragma omp parallel'
 *   - Querying the current thread's ID and the team's size
 *   - Avoiding interleaved output by serializing printf() with 'critical'
 */
int main(void)
{
    /*
     * Runtime thread limit / default:
     *   omp_get_max_threads() is an upper bound used by the runtime when it
     *   creates a team for a parallel region (unless overridden).
     */
    printf("OpenMP reports %d processors\n", omp_get_num_procs());
    printf("Max threads (runtime limit): %d\n", omp_get_max_threads());

    /*
     * PARALLEL REGION
     *   The encountering thread creates a team; each thread executes the block.
     *   Example override (disabled): '#pragma omp parallel num_threads(3)'
     */
    #pragma omp parallel /* num_threads(3) */
    {
        /*
         * Thread index within THIS parallel region: 0 .. (nthreads - 1)
         * (OpenMP terminology is typically 'thread id', not 'rank').
         */
        int tid = omp_get_thread_num();

        /* Total number of threads in this parallel region (same for all threads). */
        int nthreads = omp_get_num_threads();

        /*
         * stdout is a shared resource.
         * Serialize printing to avoid interleaved output on some runtimes/OSes.
         */
        #pragma omp critical
        {
            printf("Hello from thread %d of %d\n", tid, nthreads);
        }
    }

    /*
     * End of parallel region:
     *   All threads in the team join (implicit barrier).
     *   Execution continues with the encountering thread (here: the initial thread).
     */

    return 0;
}
