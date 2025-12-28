/*
 * OMP_Sieve_Schedule.c
 *
 * Sieve of Eratosthenes (prime search up to N) using OpenMP.
 * The program:
 *   1) Benchmarks different OpenMP scheduling strategies (static/dynamic/guided)
 *      for the marking phase of the sieve (time excludes file I/O).
 *   2) Re-runs the sieve using the fastest strategy found.
 *   3) Writes all primes <= N to an output file.
 *
 * Why schedule matters here:
 *   For each prime p, we mark multiples m = p*p, p*p+p, ... <= N.
 *   The marking loop for a given p is parallelized with #pragma omp for.
 *   Scheduling overhead and load-balance can change overall runtime,
 *   especially for large N and/or many threads.
 *
 * Build (Linux / GCC):
 *   gcc -O2 -Wall -Wextra -fopenmp OMP_Sieve_Schedule.c -o OMP_Sieve_Schedule
 *
 * Build (Windows MSYS2 MinGW-w64):
 *   gcc -O2 -Wall -Wextra -fopenmp OMP_Sieve_Schedule.c -o OMP_Sieve_Schedule.exe
 *
 * Run:
 *   ./OMP_Sieve_Schedule 10000000 primes.txt
 *
 * Optional environment controls:
 *   OMP_NUM_THREADS=8
 *   OMP_PROC_BIND=true
 *   OMP_PLACES=cores
 *
 * Notes:
 *   - Timing is for computation only. Writing the primes can dominate runtime
 *     for very large N if included in the measured interval.
 *   - The implementation uses a single parallel region to avoid repeated
 *     thread creation overhead for each p.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <omp.h>

/* Small helper for safe parsing of long from argv */
static long parse_long(const char *s)
{
    char *end = NULL;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v < 2) {
        fprintf(stderr, "Invalid N: '%s' (must be integer >= 2)\n", s);
        exit(EXIT_FAILURE);
    }
    return v;
}

/*
 * Run sieve for given N using OpenMP schedule determined by omp_set_schedule().
 *
 * Implementation approach:
 *   - is_prime[i] = 1 initially for i=0..N, then set 0 for non-primes.
 *   - Outer loop over p is executed by all threads but synchronized with a barrier
 *     each iteration to preserve correctness (all marking for p completes before
 *     any thread proceeds to p+1).
 *   - Inner marking loop uses #pragma omp for schedule(runtime), so we can
 *     choose schedule via omp_set_schedule() before calling this function.
 *
 * Returns elapsed wall time (seconds) for the computational part.
 * If out_is_prime is non-NULL, the final is_prime array is returned to caller
 * (caller must free). If NULL, the array is freed internally.
 */
static double sieve_run(long N, unsigned char **out_is_prime, long *out_count)
{
    unsigned char *is_prime = (unsigned char *)malloc((size_t)(N + 1));
    if (!is_prime) {
        fprintf(stderr, "Allocation failed for N=%ld\n", N);
        exit(EXIT_FAILURE);
    }

    /* Initialize: assume all prime, then correct */
    memset(is_prime, 1, (size_t)(N + 1));
    is_prime[0] = 0;
    is_prime[1] = 0;

    double t0 = omp_get_wtime();

    /* Single parallel region (better than creating one per prime p). */
    #pragma omp parallel
    {
        for (long p = 2; p * p <= N; ++p) {

            /*
             * Correctness point:
             * - We must not start checking/using p+1 until all threads have
             *   finished marking multiples for current p.
             * - Therefore we barrier each iteration.
             */
            if (is_prime[p]) {
                long start = p * p;          /* first multiple not already handled by smaller primes */
                long step  = p;

                #pragma omp for schedule(runtime)
                for (long m = start; m <= N; m += step) {
                    is_prime[m] = 0;
                }
            }

            #pragma omp barrier
        }
    }

    double t1 = omp_get_wtime();

    /* Count primes (sequential; can be parallelized but typically minor vs sieve). */
    long count = 0;
    for (long i = 2; i <= N; ++i) {
        if (is_prime[i]) {
            ++count;
        }
    }

    if (out_count) {
        *out_count = count;
    }

    if (out_is_prime) {
        *out_is_prime = is_prime;
    } else {
        free(is_prime);
    }

    return (t1 - t0);
}

/* Write primes to file, one per line (simple, deterministic). */
static void write_primes_to_file(const char *path, const unsigned char *is_prime, long N)
{
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "Cannot open output file '%s'\n", path);
        exit(EXIT_FAILURE);
    }

    /* Buffered output for performance */
    setvbuf(fp, NULL, _IOFBF, 1 << 20); /* 1 MiB buffer */

    for (long i = 2; i <= N; ++i) {
        if (is_prime[i]) {
            /* Each prime on its own line */
            fprintf(fp, "%ld\n", i);
        }
    }

    fclose(fp);
}

static const char *sched_name(omp_sched_t s)
{
    switch (s) {
        case omp_sched_static:  return "static";
        case omp_sched_dynamic: return "dynamic";
        case omp_sched_guided:  return "guided";
        case omp_sched_auto:    return "auto";
        default:                return "unknown";
    }
}

int main(int argc, char **argv)
{
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Usage: %s <N> [output_file]\n", argv[0]);
        fprintf(stderr, "Example: %s 10000000 primes.txt\n", argv[0]);
        return EXIT_FAILURE;
    }

    long N = parse_long(argv[1]);
    const char *out_path = (argc == 3) ? argv[2] : "primes.txt";

    /* For stable benchmarking: prevent runtime from changing thread count dynamically. */
    omp_set_dynamic(0);

    int threads = omp_get_max_threads();
    printf("N = %ld\n", N);
    printf("OpenMP max threads = %d\n\n", threads);

    /*
     * Benchmark different schedules.
     *
     * Chunk size choice:
     * - For this sieve, inner loops can be large (especially for small p).
     * - Too small chunk => higher scheduling overhead.
     * - Too large chunk => less overhead but may reduce load-balance benefits.
     *
     * We pick a moderate default chunk for dynamic/guided.
     */
    const int chunk_static  = 0;    /* implementation default */
    const int chunk_dynamic = 1024; /* reasonable baseline */
    const int chunk_guided  = 1024;

    struct {
        omp_sched_t sched;
        int chunk;
        double time_s;
        long prime_count;
    } results[3];

    results[0].sched = omp_sched_static;  results[0].chunk = chunk_static;
    results[1].sched = omp_sched_dynamic; results[1].chunk = chunk_dynamic;
    results[2].sched = omp_sched_guided;  results[2].chunk = chunk_guided;

    for (int i = 0; i < 3; ++i) {
        omp_set_schedule(results[i].sched, results[i].chunk);

        long count = 0;
        double t = sieve_run(N, NULL, &count); /* computation only */

        results[i].time_s = t;
        results[i].prime_count = count;

        printf("Schedule: %-7s  chunk: %-5d  time: %.6f s  primes: %ld\n",
               sched_name(results[i].sched),
               results[i].chunk,
               results[i].time_s,
               results[i].prime_count);
    }

    /* Choose fastest schedule for final run (and for producing the output file). */
    int best = 0;
    for (int i = 1; i < 3; ++i) {
        if (results[i].time_s < results[best].time_s) {
            best = i;
        }
    }

    printf("\nBest schedule: %s (chunk=%d), time=%.6f s\n",
           sched_name(results[best].sched),
           results[best].chunk,
           results[best].time_s);

    /* Final run using the best schedule, keeping the sieve array for file output. */
    omp_set_schedule(results[best].sched, results[best].chunk);

    unsigned char *is_prime = NULL;
    long count = 0;
    double t_final = sieve_run(N, &is_prime, &count);

    printf("Final run (for output): %.6f s, primes=%ld\n", t_final, count);

    /* File I/O is NOT included in timing above. */
    write_primes_to_file(out_path, is_prime, N);
    free(is_prime);

    printf("Primes written to: %s\n", out_path);

    return EXIT_SUCCESS;
}
