
# Compiler detection (override with: make CC=gcc)
CC ?= gcc

# Common flags
CFLAGS := -O2 -Wall -Wextra -Wpedantic -g

# OpenMP flags: gcc and clang both accept -fopenmp (with libomp installed for clang)
OMPFLAGS := -fopenmp

# Output root
BIN := bin

# Source groups
BASICS := \
  src/01_basics/omp_data_sharing.c \
  src/01_basics/omp_hello.c \
  src/01_basics/omp_parallel_for.c \
  src/01_basics/omp_reduction_fp.c \
  src/01_basics/omp_reduction.c

SYNC := \
  src/02_synchronization/omp_atomic_vs_critical.c \
  src/02_synchronization/omp_barrier.c \
  src/02_synchronization/omp_printf_interleaving.c

SCHED := \
  src/03_scheduling/omp_schedule_demo_chunks.c \
  src/03_scheduling/omp_schedule_demo.c \
  src/03_scheduling/omp_schedule_profile_used_threads.c \
  src/03_scheduling/omp_schedule_profile.c

PERF := \
  src/04_performance/omp_false_sharing_array.c \
  src/04_performance/omp_false_sharing.c \
  src/04_performance/omp_timing_reduce_max.c \
  src/04_performance/omp_timing.c

ADV := \
  src/05_advanced/omp_reduction_fp_compensated.c \
  src/05_advanced/omp_reduction_fp_pairwise.c \
  src/05_advanced/omp_simd_intro.c \
  src/05_advanced/omp_tasks_depend.c \
  src/05_advanced/omp_tasks_intro.c \
  src/05_advanced/omp_tasks_pipeline_gantt.c \
  src/05_advanced/omp_tasks_pipeline_overlap.c

SRCS := $(BASICS) $(SYNC) $(SCHED) $(PERF) $(ADV)

# Map each source to a bin path without extension and with forward slashes
# Replace src/<group>/<name>.c -> bin/<group>/<name>
EXES := $(patsubst src/%.c,$(BIN)/%,$(SRCS))

.PHONY: all clean dirs
all: dirs $(EXES)

dirs:
    mkdir -p $(BIN)/01_basics $(BIN)/02_synchronization $(BIN)/03_scheduling $(BIN)/04_performance $(BIN)/05_advanced

# Generic build rule
$(BIN)/%: src/%.c
    $(CC) $(CFLAGS) $(OMPFLAGS) $< -o $@

clean:
    $(RM) -r $(BIN)
