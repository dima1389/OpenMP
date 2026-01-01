/*
 * File:        omp_<topic>.c
 *
 * Purpose:
 *   Demonstrates <specific OpenMP concept> in a minimal and observable way.
 *
 * Key concepts:
 *   - <OpenMP directive or rule>
 *   - <Synchronization / data-sharing concept>
 *
 * OpenMP features used:
 *   Directives:
 *     - parallel
 *     - for
 *     - barrier / critical / atomic (as applicable)
 *
 *   Runtime library:
 *     - omp_get_thread_num()
 *     - omp_get_num_threads()
 *     - omp_get_wtime() (if used)
 *
 * Compilation:
 *   gcc -O2 -Wall -Wextra -Wpedantic -g -fopenmp omp_<topic>.c -o omp_<topic>
 *
 * Execution:
 *   ./omp_<topic>
 *
 * Notes:
 *   - Output order is nondeterministic unless explicitly synchronized.
 *   - This example is intended for educational purposes.
 *
 * References:
 *   - OpenMP API Specification (OpenMP ARB)
 *   - Compiler OpenMP runtime documentation
 */

/*
 * File:        <filename>
 * Purpose:     <1–2 sentences: what this program demonstrates>
 *
 * Key concepts:
 *   - <OpenMP directive or concept>
 *   - <OpenMP runtime call or concept>
 *
 * OpenMP used:
 *   - Directives: parallel, for, critical, barrier, ...
 *   - Runtime:    omp_get_thread_num(), omp_get_num_threads(), ...
 *
 * Build (GCC/Clang):
 *   gcc -O2 -Wall -Wextra -Wpedantic -g -fopenmp <file>.c -o <file>
 *
 * Run:
 *   ./<file> [args]
 *
 * Notes:
 *   - <nondeterminism / expected behavior>
 *
 * References:
 *   - OpenMP API Specification (version noted): <name only; link in README>
 *   - Compiler/runtime docs (libgomp / libomp) as relevant
 */
