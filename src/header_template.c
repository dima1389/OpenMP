/*
 * File:        omp_<topic>.c
 *
 * Purpose:
 *   <One-line statement of what this example demonstrates.>
 *
 * Description:
 *   <Short paragraph describing the educational intent and what is observable/measurable.>
 *   This example focuses on <concept> and illustrates <behavior> under <conditions>.
 *
 * Key concepts:
 *   - <OpenMP construct / clause semantics>
 *   - <Data-sharing rule: shared/private/firstprivate/default(none)>
 *   - <Synchronization / ordering / memory consistency aspect>
 *   - <Performance consideration: scheduling, overhead, locality, false sharing, etc.>
 *
 * Algorithm / workflow (high level):
 *   1) <Initialize inputs / data>
 *   2) <Enter OpenMP region(s) and perform parallel work>
 *   3) <Synchronize and/or combine partial results>
 *   4) <Report results and/or timing>
 *
 * OpenMP features used (list only those actually used in this file):
 *   Directives / constructs:
 *     - <e.g., parallel, for, simd, single, task, barrier, critical, atomic, reduction>
 *   Runtime library calls:
 *     - <e.g., omp_get_thread_num(), omp_get_num_threads(), omp_get_wtime(), ...>
 *
 * MPI features used (list only those actually used in this file):
 *   - None (OpenMP-only example)
 *
 * Compilation:
 *   GCC:
 *     gcc -O2 -Wall -Wextra -Wpedantic -g -fopenmp omp_<topic>.c -o omp_<topic>
 *   Clang:
 *     clang -O2 -Wall -Wextra -Wpedantic -g -fopenmp omp_<topic>.c -o omp_<topic>
 *
 * Execution:
 *   ./omp_<topic> [args]
 *   (Optional) OMP_NUM_THREADS=<N> ./omp_<topic> [args]
 *
 * Inputs:
 *   - Command-line arguments: <describe argv usage or "None">
 *   - Environment variables (optional): OMP_NUM_THREADS, OMP_SCHEDULE, OMP_PROC_BIND, OMP_PLACES
 *
 * References:
 *   - OpenMP API Specification (OpenMP ARB): <relevant construct/section>
 *   - Compiler runtime docs: GCC libgomp / LLVM libomp
 */
