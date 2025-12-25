/*
 * Recommended environment alignment:
 *   set "PATH=C:\msys64\mingw64\bin;%PATH%"
 *
 * Compilation:
 *   gcc -fopenmp -Wall OMP_Parallel_Sum.c -o OMP_Parallel_Sum
 *
 * Example run:
 *   OMP_Parallel_Sum 4
 */

/*
 * Standard C header for basic input/output.
 * Used here for printf() and scanf().
 */
#include <stdio.h>

/*
 * Standard C library header.
 * Used here for strtol(), which converts a string to an integer.
 */
#include <stdlib.h>

/*
 * OpenMP header.
 * Provides OpenMP runtime functions (omp_get_wtime, etc.)
 * and enables OpenMP-related constructs via pragmas.
 */
#include <omp.h>

/*
 * Program entry point.
 *
 * argc  = number of command-line arguments
 * argv  = array of command-line argument strings
 *
 * This program expects:
 *   argv[1] = number of OpenMP threads to use (tc)
 *
 * Example run:
 *   ./sum_omp 4
 *
 * Meaning: use 4 threads for the parallel loop.
 */
int main(int argc, char *argv[])
{
    /*
     * Convert the first command-line argument (argv[1]) to an integer.
     *
     * strtol(string, endptr, base) parses a string to a long integer.
     * Here:
     *   - argv[1] is the string to parse (e.g., "4")
     *   - NULL means we don't need the pointer to where parsing stopped
     *   - 10 means base-10 (decimal)
     *
     * The result is stored in tc ("thread count").
     *
     * IMPORTANT PRACTICAL NOTE:
     * This code assumes argv[1] exists. If the user does not provide it,
     * accessing argv[1] is undefined behavior (likely crash).
     */
    int tc = (int)strtol(argv[1], NULL, 10);

    /*
     * n is the numeric limit for the summation.
     * The program will compute:
     *   sum = 1 + 2 + 3 + ... + (int)n
     *
     * n is read as a double, then later cast to int for the loop bound.
     * (So if the user enters 10.9, it becomes 10 in the loop.)
     */
    double n;

    /*
     * sum will hold the final result of the summation.
     *
     * WARNING (important for OpenMP understanding):
     * In a parallel program, multiple threads might try to update "sum"
     * at the same time, which would cause a data race and wrong results.
     *
     * This program prevents that problem using:
     *   reduction(+:sum)
     */
    double sum = 0.0;

    /*
     * Ask the user for the number n.
     */
    printf("Number: ");
    scanf("%lf", &n);

    /*
     * omp_get_wtime()
     *
     * Returns a wall-clock time value (in seconds) as a double.
     * "Wall-clock time" means real elapsed time (like a stopwatch),
     * not CPU cycles or per-thread CPU time.
     *
     * We take a timestamp before the computation to measure runtime.
     */
    double s = omp_get_wtime();

    /*
     * OPENMP PARALLEL FOR WITH REDUCTION
     *
     * #pragma omp parallel for num_threads(tc) reduction(+:sum)
     *
     * This single directive does multiple things:
     *
     * 1) parallel for
     *    - The loop iterations are divided among multiple threads.
     *    - Each thread executes a subset of the iteration range.
     *
     * 2) num_threads(tc)
     *    - Requests that the OpenMP runtime uses exactly tc threads
     *      for this parallel region.
     *
     * 3) reduction(+:sum)
     *    - Solves the "multiple threads updating sum" problem safely.
     *
     *    How reduction works conceptually:
     *      - Each thread gets its own PRIVATE copy of "sum"
     *        (initialized to 0 for the + operation).
     *      - Each thread accumulates into its private sum independently.
     *      - At the end of the loop, OpenMP combines all private sums
     *        into the single shared variable "sum" using the + operator.
     *
     * Without reduction, this code would typically produce incorrect
     * results due to a data race.
     */
#pragma omp parallel for num_threads(tc) reduction(+:sum)
    for (int i = 1; i <= (int)n; i++)
    {
        /*
         * Each iteration adds i to sum.
         *
         * Because of the reduction clause:
         *   - "sum" inside the loop is thread-private (per thread)
         *   - This update is safe and race-free
         */
        sum += (double)i;
    }

    /*
     * Stop timing:
     *   current_time - start_time = elapsed time in seconds
     */
    s = omp_get_wtime() - s;

    /*
     * Print results.
     *
     * After the parallel loop completes, "sum" contains the combined
     * result from all threads due to reduction.
     */
    printf("\nSum is %lf\n", sum);
    printf("Executed for %lf s\n", s);

    /*
     * Return 0 for successful program termination.
     */
    return 0;
}
