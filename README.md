# OpenMP Programming in C — Systematic University Practice Guide

## 1. Purpose and Scope

This repository provides a **systematic, example-driven introduction to OpenMP programming in the C language**.
It is intended for university-level courses in parallel programming, computer architecture, and high-performance computing.

The project focuses on:
- Correct mental models of shared-memory parallelism
- Observable runtime behavior of OpenMP programs
- Precise use of OpenMP directives and runtime library calls
- Clear separation between **synchronization**, **ordering**, and **mutual exclusion**
- Reproducible build and execution procedures

All examples are deliberately small, self-contained, and pedagogically annotated.

---

## 2. Target Environment

- Language standard: **C99 / C11**
- Parallel programming model: **OpenMP**
- Compilers:
  - GCC (libgomp)
  - Clang/LLVM (libomp)
  - Microsoft Visual C (OpenMP subset, version-dependent)

---

## 3. Repository Structure

openmp-guide/
├── src/
│ ├── 01_basics/ # Parallel regions, thread IDs, data sharing
│ ├── 02_synchronization/ # barrier, critical, atomic, ordering effects
│ ├── 03_scheduling/ # static, dynamic, guided scheduling
│ ├── 04_performance/ # timing, false sharing, scalability
│ └── 05_advanced/ # tasks, SIMD, nested parallelism
│
├── docs/
│ ├── 00_learning_path.md
│ ├── build_linux.md
│ ├── build_windows.md
│ ├── runtime_env.md
│ └── openmp_reference.md
│
├── scripts/
│ ├── build_gcc.sh
│ └── build_msys2.cmd
│
└── README.md

---

## 4. Learning Philosophy

This project emphasizes:
- **Conceptual correctness before performance**
- Explicit discussion of **nondeterminism**
- Controlled demonstrations of race conditions and their mitigation
- Direct alignment with the official OpenMP specification

Each example answers three questions:
1. What OpenMP concept is demonstrated?
2. What behavior should be observed at runtime?
3. Why does that behavior occur?

---

## 5. Compilation (example)

```bash
gcc -O2 -Wall -Wextra -Wpedantic -g -fopenmp omp_example.c -o omp_example
```

## 6. References

- OpenMP Architecture Review Board — Official Specification
- GCC libgomp documentation
- LLVM/Clang OpenMP runtime documentation
