/*
 * File:        omp_tasks_intro.c
 *
 * Purpose:
 *   Introductory example of OpenMP tasking in C.
 *
 *   This program demonstrates:
 *     1) Creating tasks inside a single region so they are generated once
 *     2) Executing tasks by the thread team in the surrounding parallel region
 *     3) Using taskwait to wait for completion of child tasks
 *     4) Using taskgroup to define a completion scope for a set of tasks
 *
 *   The example uses a recursive Fibonacci computation as a didactic task graph:
 *   each recursive call spawns tasks for its children above a configurable cutoff.
 *
 * Key concepts:
 *   - Task parallelism vs loop parallelism
 *   - #pragma omp task and task creation
 *   - Task scheduling is managed by the OpenMP runtime (work stealing is common)
 *   - Synchronization:
 *       - taskwait: waits for direct child tasks in the current task
 *       - taskgroup: waits for all tasks generated in the group (transitively)
 *   - Granularity control: cutoff threshold to avoid excessive task overhead
 *
 * OpenMP features used:
 *   Directives:
 *     - parallel
 *     - single
 *     - task
 *     - taskwait
 *     - taskgroup
 *
 *   Runtime library:
 *     - omp_get_max_threads()
 *     - omp_get_wtime()
 *
 * Compilation (GCC/Clang):
 *   gcc -O2 -Wall -Wextra -Wpedantic -g -fopenmp omp_tasks_intro.c -o omp_tasks_intro
 *
 * Execution:
 *   ./omp_tasks_intro [n] [cutoff]
 *
 *   Arguments:
 *     n      : Fibonacci index to compute (default: 40)
 *     cutoff : recursion depth control; tasks are spawned only for n > cutoff (default: 20)
 *
 * Examples:
 *   ./omp_tasks_intro 40 20
 *   OMP_NUM_THREADS=8 ./omp_tasks_intro 42 24
 *
 * Notes:
 *   - Fibonacci is intentionally used for its branching structure; it is not the
 *     fastest way to compute Fibonacci numbers.
 *   - Task overhead can dominate if cutoff is too small (too many tiny tasks).
 *   - This example demonstrates correctness and task concepts, not peak performance.
 *
 * References:
 *   - OpenMP API Specification (OpenMP ARB): task, taskwait, taskgroup
 */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <omp.h>

/* Parse a positive int from argv with basic validation. */
static int parse_int_or_default(int argc, char *argv[], int index, int default_value)
{
    if (argc <= index) {
        return default_value;
    }

    errno = 0;
    char *end = NULL;
    long v = strtol(argv[index], &end, 10);

    if (errno != 0 || end == argv[index] || *end != '\0' || v <= 0 || v > 1000000L) {
        fprintf(stderr, "Invalid integer value at argv[%d]: '%s'\n", index, argv[index]);
        fprintf(stderr, "Usage: %s [n] [cutoff]\n", argv[0]);
        exit(1);
    }

    return (int)v;
}

/*
 * fib_task(n, cutoff) computes Fibonacci(n) using OpenMP tasks for parallelism.
 *
 * Rule:
 *   - If n is small, compute serially to avoid task overhead.
 *   - If n is large (n > cutoff), spawn tasks for the two subproblems and wait.
 *
 * Technical note:
 *   - The "shared(x)" clauses are required because x is a local variable of the
 *     current task context; we want child tasks to write their results into it.
 */
static long long fib_task(int n, int cutoff)
{
    if (n < 2) {
        return (long long)n;
    }

    /*
     * For small n, do serial recursion. This improves performance and avoids
     * creating a huge number of fine-grained tasks.
     */
    if (n <= cutoff) {
        return fib_task(n - 1, cutoff) + fib_task(n - 2, cutoff);
    }

    long long x = 0;
    long long y = 0;

    /*
     * taskgroup:
     *   Ensures that all tasks created in this lexical region complete before exit,
     *   including tasks spawned by descendant tasks (transitively).
     *
     * This provides a clean, structured way to express completion.
     */
    #pragma omp taskgroup
    {
        #pragma omp task default(none) shared(x) firstprivate(n, cutoff)
        {
            x = fib_task(n - 1, cutoff);
        }

        #pragma omp task default(none) shared(y) firstprivate(n, cutoff)
        {
            y = fib_task(n - 2, cutoff);
        }

        /*
         * No explicit taskwait is required here because taskgroup already provides
         * the completion guarantee for tasks created within the group.
         */
    }

    return x + y;
}

int main(int argc, char *argv[])
{
    const int default_n = 40;
    const int default_cutoff = 20;

    int n = parse_int_or_default(argc, argv, 1, default_n);
    int cutoff = parse_int_or_default(argc, argv, 2, default_cutoff);

    if (cutoff < 2) {
        fprintf(stderr, "Cutoff should be >= 2 for meaningful task granularity control.\n");
        return 1;
    }

    printf("OpenMP tasks introduction\n");
    printf("Compute Fibonacci(n) with task parallelism\n");
    printf("n = %d, cutoff = %d\n", n, cutoff);
    printf("Max threads available: %d\n\n", omp_get_max_threads());

    long long result = 0;

    double t0 = omp_get_wtime();

    /*
     * Parallel region creates the worker threads.
     * The single region ensures only one thread generates the initial task tree.
     */
    #pragma omp parallel default(none) shared(result, n, cutoff)
    {
        #pragma omp single
        {
            result = fib_task(n, cutoff);
        }
    }

    double t1 = omp_get_wtime();

    printf("Result: Fibonacci(%d) = %lld\n", n, result);
    printf("Elapsed time: %.6f s\n\n", t1 - t0);

    printf("Interpretation:\n");
    printf("  - Tasks allow irregular and recursive parallelism that does not fit a simple\n");
    printf("    parallel-for loop.\n");
    printf("  - The cutoff parameter controls task granularity; too small => many tasks and\n");
    printf("    high overhead; too large => insufficient parallelism.\n");
    printf("  - The OpenMP runtime schedules tasks across threads dynamically.\n");

    return 0;
}
