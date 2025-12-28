/*
 * File:        omp_tasks_pipeline_overlap.c
 *
 * Purpose:
 *   Demonstrates observable pipeline overlap using OpenMP task dependencies.
 *
 *   This program extends omp_tasks_depend.c by adding lightweight instrumentation:
 *     - per-stage start/end timestamps (relative to program start)
 *     - a concise event log printed after execution
 *
 *   The goal is to make *overlap* visible:
 *     - different pipeline items can be in different stages concurrently
 *     - ordering within an item is enforced by depend() clauses
 *
 *   Pipeline per item:
 *     Stage A: produce   (depend(out: a[i]))
 *     Stage B: transform (depend(in: a[i]) depend(out: b[i]))
 *     Stage C: consume   (depend(in: b[i]))
 *
 * Key concepts:
 *   - Task dependency graphs (DAGs)
 *   - Pipeline parallelism and overlap
 *   - Instrumentation: measuring and reporting concurrency behavior
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
 *       omp_tasks_pipeline_overlap.c -o omp_tasks_pipeline_overlap
 *
 * Execution:
 *   ./omp_tasks_pipeline_overlap [items] [verbosity]
 *
 *   Arguments:
 *     items     : number of pipeline items (default: 8)
 *     verbosity : 0 = summary only, 1 = full event log (default: 1)
 *
 * Notes:
 *   - The event log is printed after the parallel region completes to avoid I/O
 *     serialization effects during execution.
 *   - Timings are relative and machine-dependent; they are used for qualitative
 *     overlap visualization, not benchmarking.
 *
 * References:
 *   - OpenMP API Specification (OpenMP ARB): task depend clause
 */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <omp.h>

/* ---------- argument parsing helpers ---------- */

static int parse_int_or_default(int argc, char *argv[], int index, int default_value)
{
    if (argc <= index) {
        return default_value;
    }

    errno = 0;
    char *end = NULL;
    long v = strtol(argv[index], &end, 10);

    if (errno != 0 || end == argv[index] || *end != '\0') {
        fprintf(stderr, "Invalid integer value at argv[%d]: '%s'\n", index, argv[index]);
        fprintf(stderr, "Usage: %s [items] [verbosity]\n", argv[0]);
        exit(1);
    }

    return (int)v;
}

/* ---------- deterministic synthetic work ---------- */

static void burn_work(int cost)
{
    volatile double acc = 0.0;
    for (int i = 0; i < cost * 120000; ++i) {
        acc += (double)i * 1e-7;
    }
}

/* ---------- event logging ---------- */

typedef struct {
    int item;
    char stage;          /* 'A', 'B', 'C' */
    int tid;
    double t_start;      /* seconds since t0 */
    double t_end;        /* seconds since t0 */
} event_t;

static int stage_index(char s)
{
    return (s == 'A') ? 0 : (s == 'B') ? 1 : 2;
}

static const char *stage_name(char s)
{
    return (s == 'A') ? "A (produce)" : (s == 'B') ? "B (transform)" : "C (consume)";
}

/*
 * Compare events by start time, then by item, then by stage.
 * Used only for sorting before printing; execution does not depend on order.
 */
static int cmp_event(const void *pa, const void *pb)
{
    const event_t *a = (const event_t *)pa;
    const event_t *b = (const event_t *)pb;

    if (a->t_start < b->t_start) return -1;
    if (a->t_start > b->t_start) return  1;

    if (a->item < b->item) return -1;
    if (a->item > b->item) return  1;

    int sa = stage_index(a->stage);
    int sb = stage_index(b->stage);
    return (sa < sb) ? -1 : (sa > sb) ? 1 : 0;
}

int main(int argc, char *argv[])
{
    const int default_items = 8;
    const int default_verbosity = 1;

    int items = parse_int_or_default(argc, argv, 1, default_items);
    int verbosity = parse_int_or_default(argc, argv, 2, default_verbosity);

    if (items <= 0) {
        fprintf(stderr, "items must be > 0\n");
        return 1;
    }

    if (verbosity != 0 && verbosity != 1) {
        fprintf(stderr, "verbosity must be 0 or 1\n");
        return 1;
    }

    printf("OpenMP pipeline overlap demonstration (tasks + depend)\n");
    printf("items = %d, verbosity = %d\n", items, verbosity);
    printf("Max threads available: %d\n\n", omp_get_max_threads());

    /*
     * Dependency tokens:
     * Use element addresses token_a[i] and token_b[i] to express per-item dependencies.
     */
    int *token_a = (int *)calloc((size_t)items, sizeof(int));
    int *token_b = (int *)calloc((size_t)items, sizeof(int));
    if (token_a == NULL || token_b == NULL) {
        fprintf(stderr, "Allocation failure for dependency tokens.\n");
        return 1;
    }

    /*
     * Event log:
     * 3 stages per item => 3 * items events.
     */
    const int total_events = 3 * items;
    event_t *events = (event_t *)calloc((size_t)total_events, sizeof(event_t));
    if (events == NULL) {
        fprintf(stderr, "Allocation failure for event log.\n");
        free(token_a);
        free(token_b);
        return 1;
    }

    double t0 = omp_get_wtime();

    #pragma omp parallel default(none) shared(items, token_a, token_b, events, t0)
    {
        #pragma omp single
        {
            for (int i = 0; i < items; ++i) {

                /* Stage A: produce */
                #pragma omp task default(none) firstprivate(i) shared(token_a, events, t0) \
                                 depend(out: token_a[i])
                {
                    int tid = omp_get_thread_num();
                    int idx = 3 * i + 0;

                    events[idx].item = i;
                    events[idx].stage = 'A';
                    events[idx].tid = tid;
                    events[idx].t_start = omp_get_wtime() - t0;

                    burn_work(2);
                    token_a[i] = i;

                    events[idx].t_end = omp_get_wtime() - t0;
                }

                /* Stage B: transform */
                #pragma omp task default(none) firstprivate(i) shared(token_a, token_b, events, t0) \
                                 depend(in: token_a[i]) depend(out: token_b[i])
                {
                    int tid = omp_get_thread_num();
                    int idx = 3 * i + 1;

                    events[idx].item = i;
                    events[idx].stage = 'B';
                    events[idx].tid = tid;
                    events[idx].t_start = omp_get_wtime() - t0;

                    burn_work(3);
                    token_b[i] = token_a[i] * 2;

                    events[idx].t_end = omp_get_wtime() - t0;
                }

                /* Stage C: consume */
                #pragma omp task default(none) firstprivate(i) shared(token_b, events, t0) \
                                 depend(in: token_b[i])
                {
                    int tid = omp_get_thread_num();
                    int idx = 3 * i + 2;

                    events[idx].item = i;
                    events[idx].stage = 'C';
                    events[idx].tid = tid;
                    events[idx].t_start = omp_get_wtime() - t0;

                    burn_work(1);
                    (void)token_b[i]; /* consumption (value not used further) */

                    events[idx].t_end = omp_get_wtime() - t0;
                }
            }
        }
    }

    double t_end = omp_get_wtime() - t0;

    /* Sort events by start time for readable printing */
    qsort(events, (size_t)total_events, sizeof(event_t), cmp_event);

    printf("Total elapsed time: %.6f s\n\n", t_end);

    /*
     * Summary: show completion by item and demonstrate ordering within each item.
     * (We do not enforce stable stage times; we only show that A precedes B precedes C.)
     */
    if (verbosity == 0) {
        printf("Summary (per item):\n");
        printf("Item | A_end    | B_end    | C_end\n");
        printf("-----+----------+----------+----------\n");

        for (int i = 0; i < items; ++i) {
            double a_end = 0.0, b_end = 0.0, c_end = 0.0;

            for (int k = 0; k < total_events; ++k) {
                if (events[k].item != i) continue;
                if (events[k].stage == 'A') a_end = events[k].t_end;
                if (events[k].stage == 'B') b_end = events[k].t_end;
                if (events[k].stage == 'C') c_end = events[k].t_end;
            }

            printf("%4d | %8.4f | %8.4f | %8.4f\n", i, a_end, b_end, c_end);
        }

        printf("\n");
    } else {
        printf("Event log (sorted by start time):\n");
        printf("Start    End      Dur      TID  Item  Stage\n");
        printf("-------- -------- -------- ---- ----- ----------------\n");

        for (int k = 0; k < total_events; ++k) {
            double dur = events[k].t_end - events[k].t_start;

            printf("%8.4f %8.4f %8.4f %4d %5d %s\n",
                   events[k].t_start,
                   events[k].t_end,
                   dur,
                   events[k].tid,
                   events[k].item,
                   stage_name(events[k].stage));
        }

        printf("\n");
    }

    printf("Interpretation:\n");
    printf("  - Overlap is visible when events from different items interleave in time.\n");
    printf("  - Within a single item, depend() enforces A -> B -> C ordering.\n");
    printf("  - The runtime schedules tasks as soon as their dependencies are satisfied.\n");

    free(token_a);
    free(token_b);
    free(events);

    return 0;
}
