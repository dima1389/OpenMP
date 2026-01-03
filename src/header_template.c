/*
 * File:        omp_<topic>.c
 *
 * Purpose:
 *   Demonstrates <specific OpenMP concept> in a minimal and observable way.
 *
 *   This program shows:
 *     - <specific behavior or pattern>
 *     - <additional behavior or pattern>
 *     - <additional behavior or pattern>
 *
 *   <Optional: detailed explanation of the educational intent and
 *    relationship to other examples in the project>
 *
 * Key concepts:
 *   - <OpenMP directive or rule>
 *   - <Synchronization / data-sharing concept>
 *   - <Performance or behavior consideration>
 *   - <Additional concept>
 *
 * OpenMP features used:
 *   Directives (list only those actually used in this file):
 *     - parallel
 *     - for
 *     - simd
 *     - parallel for simd
 *     - barrier
 *     - critical
 *     - atomic
 *     - reduction
 *     - single
 *     - task
 *     - taskwait
 *     - taskgroup
 *     - depend
 *
 *   Runtime library (list only those actually called in this file):
 *     - omp_get_thread_num()
 *     - omp_get_num_threads()
 *     - omp_get_max_threads()
 *     - omp_get_wtime()
 *     - omp_get_schedule()
 *
 * Compilation (GCC / Clang):
 *   gcc -O2 -Wall -Wextra -Wpedantic -g -fopenmp \
 *       omp_<topic>.c -o omp_<topic>
 *
 *   For SIMD-intensive code:
 *   gcc -O3 -march=native -ffast-math -Wall -Wextra -Wpedantic -g -fopenmp \
 *       omp_<topic>.c -o omp_<topic>
 *   Note: -ffast-math relaxes IEEE 754 compliance and may affect numerical accuracy.
 *         Use only when approximate results are acceptable (e.g., graphics, physics simulations).
 *         For intermediate options: -fno-signed-zeros, -fno-trapping-math, or -fno-math-errno.
 *
 * Execution:
 *   ./omp_<topic>
 *
 *   With arguments:
 *   ./omp_<topic> [N] [options]
 *
 *   Arguments:
 *     N       : <description of first argument> (default: <value>)
 *     options : <description of additional arguments> (default: <value>)
 *
 *   Optional environment variables:
 *     export OMP_NUM_THREADS=4
 *     export OMP_SCHEDULE=<type>[,chunk_size]  # chunk_size for static/dynamic/guided
 *     export OMP_PROC_BIND=true
 *     export OMP_PLACES=cores
 *
 * Examples:
 *   ./omp_<topic>
 *   ./omp_<topic> 1000000
 *   OMP_NUM_THREADS=8 ./omp_<topic> 1000000 5
 *
 * Notes:
 *   - Output order is nondeterministic unless explicitly synchronized.
 *   - This example is intended for educational purposes.
 *   - <Additional note about behavior or expected output>
 *   - <Additional note about compilation or portability>
 *
 * Portability note (Windows/MinGW):
 *   - Thread affinity APIs may differ; OMP_PROC_BIND behavior varies by runtime
 *   - Use _aligned_malloc/_aligned_free for aligned memory allocation
 *   - Link against libgomp or libomp depending on compiler (GCC vs Clang)
 *
 * References:
 *   - OpenMP API Specification (OpenMP ARB): <specific sections or constructs>
 *   - Compiler OpenMP runtime documentation (libgomp / libomp)
 *   - <Additional references if applicable>
 */
