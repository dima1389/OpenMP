/*
 * Recommended environment alignment:
 *   set "PATH=C:\msys64\mingw64\bin;%PATH%"
 *
 * Compilation:
 *   gcc -fopenmp -Wall OMP_Hello.c -o OMP_Hello
 *
 * Execution:
 *   OMP_Hello
 */

/*
 * Standard C header that provides input/output functionality.
 * In this program it is used for the printf() function,
 * which prints text to the standard output (console).
 */
#include <stdio.h>

/*
 * OpenMP header file.
 *
 * This header declares all OpenMP runtime library functions,
 * such as:
 *   - omp_get_thread_num()
 *   - omp_get_num_threads()
 *
 * It must be included in every C/C++ program that uses OpenMP.
 */
#include <omp.h>

/*
 * The main entry point of the program.
 *
 * argc and argv are standard C parameters for command-line arguments,
 * but they are not used in this example.
 */
int main(int argc, char *argv[])
{
    /*
     * OpenMP PARALLEL DIRECTIVE
     *
     * #pragma omp parallel
     *
     * This directive tells the compiler:
     *   "Execute the following block of code in parallel,
     *    using multiple threads."
     *
     * When the program reaches this directive:
     *   - A team of threads is created.
     *   - Each thread executes the code inside the braces { }.
     *
     * The number of threads is determined by:
     *   - Environment variable OMP_NUM_THREADS, OR
     *   - Runtime defaults, OR
     *   - A clause such as num_threads(N).
     *
     * Example (commented out below):
     *   num_threads(3) → forces exactly 3 threads.
     */
#pragma omp parallel /* num_threads(3) */
    {
        /*
         * omp_get_thread_num()
         *
         * Returns the ID (index) of the current thread
         * within the current parallel region.
         *
         * Thread IDs:
         *   - Start at 0
         *   - End at (number_of_threads - 1)
         *
         * Each thread gets a UNIQUE ID.
         */
        int trank = omp_get_thread_num();

        /*
         * omp_get_num_threads()
         *
         * Returns the TOTAL number of threads
         * participating in the current parallel region.
         *
         * This value is the SAME for all threads.
         */
        int tc = omp_get_num_threads();

        /*
         * Print a message from each thread.
         *
         * Because this printf() is inside a parallel region:
         *   - Every thread executes it once.
         *   - Output lines may appear in ANY order.
         *
         * Example output (order is not guaranteed):
         *   Hello from thread 0 of 4
         *   Hello from thread 2 of 4
         *   Hello from thread 1 of 4
         *   Hello from thread 3 of 4
         *
         * The order depends on thread scheduling by the OS.
         */
        printf(
            "Hello from thread %d of %d\n",
            trank,
            tc
        );
    }

    /*
     * After the parallel region ends:
     *   - All threads are synchronized.
     *   - Only the original master thread continues execution.
     *
     * This is an implicit barrier at the end of the parallel block.
     */

    /*
     * Return 0 to indicate successful program termination.
     */
    return 0;
}
