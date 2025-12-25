/*
 * Toolchain and runtime DLL consistency notice (Windows / MinGW-w64):
 *
 * This example uses GCC with OpenMP support under MSYS2 (MinGW-w64).
 * On Windows, the compiler front-end (cc1plus.exe) dynamically loads
 * runtime libraries (e.g. zlib1.dll) from directories listed in PATH.
 * If multiple MSYS2 environments (mingw64, ucrt64) or other toolchains
 * (Git, Anaconda, Cygwin) are present on PATH, an incompatible zlib1.dll
 * may be loaded at runtime, resulting in errors such as:
 *
 *   "cc1plus.exe – Entry Point Not Found: crc32_combine"
 *
 * To avoid DLL mismatches, ensure that a single MinGW-w64 environment
 * is used and that its bin directory appears first in PATH.
 *
 * Recommended environment alignment:
 *   set "PATH=C:\msys64\mingw64\bin;%PATH%"
 *
 * Compilation:
 *   gcc -fopenmp -Wall OMP_Barrier.c -o OMP_Barrier.exe
 *
 * Execution:
 *   OMP_Barrier.exe
 *
 * Notes:
 * - Do not mix mingw64 and ucrt64 toolchains in the same CMD session.
 * - Prefer the MSYS2 “MinGW x64” shell for a preconfigured environment.
 */

/*
 * Standard C header providing input/output functions.
 * In this example, printf() is used to print text to the console.
 */
#include <stdio.h>

/*
 * OpenMP header file.
 *
 * This header makes all OpenMP runtime functions and constructs
 * available to the program, including:
 *   - omp_get_thread_num()
 *   - synchronization constructs such as barriers
 */
#include <omp.h>

/*
 * Program entry point.
 *
 * argc and argv represent command-line arguments,
 * but they are not used in this example.
 */
int main(int argc, char *argv[])
{
    /*
     * OPENMP PARALLEL REGION
     *
     * This directive creates a team of threads.
     * Each thread executes the code inside the block { } independently.
     *
     * All threads start executing the code in this block at (almost)
     * the same time.
     */
#pragma omp parallel
    {
        /*
         * FIRST PRINTF STATEMENT (BEFORE BARRIER)
         *
         * Each thread executes this printf() as soon as it enters
         * the parallel region.
         *
         * Because there is NO synchronization before this point:
         *   - Threads may reach this printf() at different times
         *   - Output order is NOT deterministic
         *
         * omp_get_thread_num() returns:
         *   - The ID of the current thread (0, 1, 2, ...)
         */
        printf(
            "Printf 1 of %d thread\n",
            omp_get_thread_num()
        );

        /*
         * OPENMP BARRIER
         *
         * #pragma omp barrier
         *
         * A barrier is a synchronization point.
         *
         * What it means:
         *   - Every thread MUST reach this line
         *   - No thread is allowed to continue past this point
         *     until ALL threads have arrived here
         *
         * If one thread reaches the barrier early:
         *   - It waits (is blocked)
         *
         * If another thread is slow:
         *   - All other threads wait for it
         *
         * Only when the LAST thread arrives at the barrier
         * are all threads released to continue execution.
         */
#pragma omp barrier

        /*
         * SECOND PRINTF STATEMENT (AFTER BARRIER)
         *
         * This printf() is executed only AFTER:
         *   - All threads have completed "Printf 1"
         *   - All threads have reached the barrier
         *
         * Important:
         *   - The order of "Printf 2" lines is still not guaranteed
         *   - But NONE of them can appear before ALL "Printf 1" lines
         *
         * This demonstrates how a barrier enforces a global
         * synchronization point among threads.
         */
        printf(
            "Printf 2 of %d thread\n",
            omp_get_thread_num()
        );
    }

    /*
     * End of the parallel region.
     *
     * There is an implicit barrier here:
     *   - All threads synchronize
     *   - Only the original (master) thread continues execution
     */

    /*
     * Return 0 to indicate successful program termination.
     */
    return 0;
}
