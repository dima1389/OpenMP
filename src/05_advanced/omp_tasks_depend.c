/*
 * File:        omp_tasks_depend.c
 *
 * Purpose:
 *   Demonstrates OpenMP task dependencies using the depend clause.
 *
 *   This example introduces *explicit task graphs* (DAGs) by expressing
 *   data dependencies between tasks using:
 *
 *     depend(in:    ...)
 *     depend(out:   ...)
 *     depend(inout: ...)
 *
 *   The program models a simple three-stage pipeline:
 *
 *     Stage A: produce data
 *     Stage B: transform data
 *     Stage C: consume data
 *
 *   Multiple pipeline instances are created, and OpenMP is allowed to
 *   execute tasks concurrently *only when dependencies permit*.
 *
 * Key concepts:
 *   - Task dependency graphs (DAGs)
 *   - depend(in/out/inout) semantics
 *   - Correct ordering without global synchronization
 *   - Difference between taskwait and dependency-based synchronization
 *
 * OpenMP features used:
 *   Directives:
 *     - parallel
 *     - single
 *     - task
 *     - depend
 *
 *   Runtime library:
 *     - omp_get_max_threads()
 *     - omp_get_thread_num()
 *     - omp_get_wtime()
 *
 * Compilation (GCC/Clang):
 *   gcc -O2 -Wall -Wextra -Wpedantic -g -fopenmp \
 *       omp_tasks_depend.c -o omp_tasks_depend
 *
 * Execution:
 *   ./omp_tasks_depend [N]
 *
 *   Arguments:
 *     N : number of independent pipeline items (default: 8)
 *
 * Notes:
 *   - The depend clause expresses *logical data dependencies*, not mutexes.
 *   - No explicit taskwait is required; dependencies enforce correct ordering.
 *   - This model maps naturally to producer–consumer pipelines and DAG workloads.
 *
 * References:
 *   - OpenMP API Specification (OpenMP ARB): task depend clause
 */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <omp.h>

/* Parse a positive int from argv with validation. */
static int parse_int_or_default(int argc, char *argv[], int index, int default_value)
{
    if (argc <= index) {
        return default_value;
    }

    errno = 0;
    char *end = NULL;
    long v = strtol(argv[index], &end, 10);

    if (errno != 0 || end == argv[index] || *end != '\0' || v <= 0) {
        fprintf(stderr, "Invalid integer value at argv[%d]: '%s'\n", index, argv[index]);
        fprintf(stderr, "Usage: %s [N]\n", argv[0]);
        exit(1);
    }

    return (int)v;
}

/* Simulated work for pipeline stages */
static void work(const char *label, int item, int cost)
{
    volatile double acc = 0.0;
    for (int i = 0; i < cost * 100000; ++i) {
        acc += (double)i * 1e-7;
    }

    printf("Thread %d: %s item %d\n",
           omp_get_thread_num(), label, item);
}

/* ---------- main ---------- */

int main(int argc, char *argv[])
{
    const int default_items = 8;
    int items = parse_int_or_default(argc, argv, 1, default_items);

    printf("OpenMP task dependencies demonstration\n");
    printf("Pipeline items: %d\n", items);
    printf("Max threads available: %d\n\n", omp_get_max_threads());

    /*
     * Each pipeline item has its own logical data tokens.
     * Dependencies are expressed via addresses of these tokens.
     */
    int *token_a = (int *)calloc((size_t)items, sizeof(int));
    int *token_b = (int *)calloc((size_t)items, sizeof(int));

    if (token_a == NULL || token_b == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    double t0 = omp_get_wtime();

    #pragma omp parallel default(none) shared(items, token_a, token_b)
    {
        #pragma omp single
        {
            for (int i = 0; i < items; ++i) {

                /* Stage A: produce data */
                #pragma omp task default(none) firstprivate(i) shared(token_a) \
                                 depend(out: token_a[i])
                {
                    work("Stage A (produce)", i, 2);
                    token_a[i] = i;
                }

                /* Stage B: transform data */
                #pragma omp task default(none) firstprivate(i) shared(token_a, token_b) \
                                 depend(in: token_a[i]) depend(out: token_b[i])
                {
                    work("Stage B (transform)", i, 3);
                    token_b[i] = token_a[i] * 2;
                }

                /* Stage C: consume data */
                #pragma omp task default(none) firstprivate(i) shared(token_b) \
                                 depend(in: token_b[i])
                {
                    work("Stage C (consume)", i, 1);
                }
            }
        }
    }

    double t1 = omp_get_wtime();

    printf("\nElapsed time: %.6f s\n", t1 - t0);

    printf("\nInterpretation:\n");
    printf("  - Tasks from different pipeline items may execute concurrently.\n");
    printf("  - Within a single item, Stage A -> B -> C ordering is enforced\n");
    printf("    purely by depend clauses (no barriers or taskwait).\n");
    printf("  - The runtime builds a task dependency graph (DAG) and schedules\n");
    printf("    tasks as soon as their dependencies are satisfied.\n");

    free(token_a);
    free(token_b);

    return 0;
}
