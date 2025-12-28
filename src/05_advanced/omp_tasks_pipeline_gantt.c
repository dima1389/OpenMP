/*
 * File:        omp_tasks_pipeline_gantt.c
 *
 * Purpose:
 *   Produces a simple ASCII Gantt-like visualization of OpenMP task pipeline overlap.
 *
 *   This program is a continuation of:
 *     - omp_tasks_intro.c
 *     - omp_tasks_depend.c
 *     - omp_tasks_pipeline_overlap.c
 *
 *   It creates a dependency-driven pipeline per item:
 *     A (produce) -> B (transform) -> C (consume)
 *
 *   Each stage records start/end timestamps and the executing thread ID.
 *   After execution, the program prints:
 *     1) A per-thread ASCII timeline showing where each task ran in time
 *     2) A sorted event list (optional)
 *
 *   This makes concurrency and overlap visually obvious without external tools.
 *
 * Key concepts:
 *   - OpenMP task dependencies (depend)
 *   - Pipeline parallelism and overlap
 *   - Instrumentation and visualization of execution traces
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
 *       omp_tasks_pipeline_gantt.c -o omp_tasks_pipeline_gantt
 *
 * Execution:
 *   ./omp_tasks_pipeline_gantt [items] [width] [print_events]
 *
 *   Arguments:
 *     items        : number of pipeline items (default: 8)
 *     width        : timeline width in characters (default: 80)
 *     print_events : 0 = only gantt, 1 = gantt + event list (default: 0)
 *
 * Examples:
 *   OMP_NUM_THREADS=4 ./omp_tasks_pipeline_gantt 12 100 0
 *   OMP_NUM_THREADS=8 ./omp_tasks_pipeline_gantt 16 120 1
 *
 * Notes:
 *   - This is a qualitative visualization tool, not a benchmark.
 *   - Timeline granularity depends on width and total elapsed time.
 *   - Printing is performed only after tasks complete to avoid perturbing scheduling.
 *
 * References:
 *   - OpenMP API Specification (OpenMP ARB): task depend clause
 */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <omp.h>

/* ---------- argument parsing helpers ---------- */

static int parse_int_or_default(int argc, char *argv[], int index, int def)
{
    if (argc <= index) {
        return def;
    }

    errno = 0;
    char *end = NULL;
    long v = strtol(argv[index], &end, 10);

    if (errno != 0 || end == argv[index] || *end != '\0') {
        fprintf(stderr, "Invalid integer value at argv[%d]: '%s'\n", index, argv[index]);
        fprintf(stderr, "Usage: %s [items] [width] [print_events]\n", argv[0]);
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
    char stage;      /* 'A', 'B', 'C' */
    int tid;
    double t_start;  /* seconds since t0 */
    double t_end;    /* seconds since t0 */
} event_t;

static int stage_index(char s)
{
    return (s == 'A') ? 0 : (s == 'B') ? 1 : 2;
}

static const char *stage_name(char s)
{
    return (s == 'A') ? "A (produce)" : (s == 'B') ? "B (transform)" : "C (consume)";
}

static int cmp_event_start(const void *pa, const void *pb)
{
    const event_t *a = (const event_t *)pa;
    const event_t *b = (const event_t *)pb;

    if (a->t_start < b->t_start) return -1;
    if (a->t_start > b->t_start) return  1;

    if (a->tid < b->tid) return -1;
    if (a->tid > b->tid) return  1;

    if (a->item < b->item) return -1;
    if (a->item > b->item) return  1;

    int sa = stage_index(a->stage);
    int sb = stage_index(b->stage);
    return (sa < sb) ? -1 : (sa > sb) ? 1 : 0;
}

/* Map time in [0, T] to a column in [0, width-1]. */
static int time_to_col(double t, double total, int width)
{
    if (total <= 0.0) {
        return 0;
    }
    double x = t / total;
    if (x < 0.0) x = 0.0;
    if (x > 1.0) x = 1.0;
    int col = (int)(x * (double)(width - 1));
    if (col < 0) col = 0;
    if (col >= width) col = width - 1;
    return col;
}

/*
 * Draw one event into the timeline row:
 * - fills [start_col, end_col] with a stage char
 * - writes item id digits near the start if space permits (optional)
 */
static void draw_event(char *row, int width, const event_t *e, double total)
{
    int c0 = time_to_col(e->t_start, total, width);
    int c1 = time_to_col(e->t_end, total, width);

    if (c1 < c0) {
        int tmp = c0;
        c0 = c1;
        c1 = tmp;
    }

    if (c0 == c1) {
        row[c0] = e->stage;
        return;
    }

    for (int c = c0; c <= c1 && c < width; ++c) {
        row[c] = e->stage;
    }

    /* Optional: annotate item number near start (best-effort) */
    if (c0 + 2 < width) {
        int item = e->item;
        if (item >= 0 && item <= 99) {
            row[c0] = e->stage;
            if (item >= 10) {
                row[c0 + 1] = (char)('0' + (item / 10));
                row[c0 + 2] = (char)('0' + (item % 10));
            } else {
                row[c0 + 1] = (char)('0' + item);
            }
        }
    }
}

int main(int argc, char *argv[])
{
    const int default_items = 8;
    const int default_width = 80;
    const int default_print_events = 0;

    int items = parse_int_or_default(argc, argv, 1, default_items);
    int width = parse_int_or_default(argc, argv, 2, default_width);
    int print_events = parse_int_or_default(argc, argv, 3, default_print_events);

    if (items <= 0) {
        fprintf(stderr, "items must be > 0\n");
        return 1;
    }
    if (width < 40) {
        fprintf(stderr, "width must be >= 40 for readable output\n");
        return 1;
    }
    if (print_events != 0 && print_events != 1) {
        fprintf(stderr, "print_events must be 0 or 1\n");
        return 1;
    }

    printf("OpenMP pipeline Gantt visualization (tasks + depend)\n");
    printf("items = %d, width = %d, print_events = %d\n", items, width, print_events);
    printf("Max threads available: %d\n\n", omp_get_max_threads());

    /* Dependency tokens */
    int *token_a = (int *)calloc((size_t)items, sizeof(int));
    int *token_b = (int *)calloc((size_t)items, sizeof(int));
    if (token_a == NULL || token_b == NULL) {
        fprintf(stderr, "Allocation failure for dependency tokens.\n");
        return 1;
    }

    /* 3 events per item */
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
                    (void)token_b[i];

                    events[idx].t_end = omp_get_wtime() - t0;
                }
            }
        }
    }

    double total = omp_get_wtime() - t0;

    /* Sort by start time to print events and to draw in a stable order */
    qsort(events, (size_t)total_events, sizeof(event_t), cmp_event_start);

    printf("Total elapsed time: %.6f s\n\n", total);

    /*
     * Build per-thread timeline.
     * We draw the timeline using the maximum thread id observed in the trace.
     * This corresponds to the team used for tasks in this run.
     */
    int max_tid = 0;
    for (int k = 0; k < total_events; ++k) {
        if (events[k].tid > max_tid) {
            max_tid = events[k].tid;
        }
    }
    int used_threads = max_tid + 1;

    /* Allocate timeline rows */
    char **rows = (char **)calloc((size_t)used_threads, sizeof(char *));
    if (rows == NULL) {
        fprintf(stderr, "Allocation failure for timeline rows.\n");
        free(token_a);
        free(token_b);
        free(events);
        return 1;
    }

    for (int t = 0; t < used_threads; ++t) {
        rows[t] = (char *)calloc((size_t)width + 1, sizeof(char));
        if (rows[t] == NULL) {
            fprintf(stderr, "Allocation failure for timeline row %d.\n", t);
            for (int j = 0; j < t; ++j) free(rows[j]);
            free(rows);
            free(token_a);
            free(token_b);
            free(events);
            return 1;
        }
        memset(rows[t], '.', (size_t)width);
        rows[t][width] = '\0';
    }

    /* Draw each event into its thread row */
    for (int k = 0; k < total_events; ++k) {
        int tid = events[k].tid;
        if (tid >= 0 && tid < used_threads) {
            draw_event(rows[tid], width, &events[k], total);
        }
    }

    /* Print legend and Gantt chart */
    printf("Legend:\n");
    printf("  A = produce, B = transform, C = consume\n");
    printf("  Digits after stage letter indicate item id (best-effort annotation)\n\n");

    printf("Gantt-like timeline (each row = one OpenMP thread):\n");
    printf("Time: 0");
    for (int i = 0; i < width - 10; ++i) {
        if (i == (width - 11) / 2) {
            printf("|");
        } else if (i == width - 12) {
            printf("|");
        } else {
            printf("-");
        }
    }
    printf("T=%.3fs\n", total);

    for (int t = 0; t < used_threads; ++t) {
        printf("T%02d: %s\n", t, rows[t]);
    }

    printf("\nInterpretation:\n");
    printf("  - Overlap is visible when multiple thread rows contain activity (A/B/C) at the same time.\n");
    printf("  - Within each item, A must complete before B, and B before C (depend() constraints).\n");
    printf("  - The runtime may schedule tasks on any worker thread, so stages for a given item\n");
    printf("    can appear on different rows.\n");

    if (print_events) {
        printf("\nEvent list (sorted by start time):\n");
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

    /* Cleanup */
    for (int t = 0; t < used_threads; ++t) {
        free(rows[t]);
    }
    free(rows);

    free(token_a);
    free(token_b);
    free(events);

    return 0;
}
