@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem ============================================================================
rem  build.bat (project root)
rem
rem  Purpose:
rem    Compile all OpenMP C examples into the bin\ folder, preserving the same
rem    subfolder structure as src\.
rem
rem  Usage:
rem    build.bat                 -> build everything
rem    set CC=clang && build.bat -> override compiler (default: gcc)
rem
rem  Notes:
rem    - Requires a compiler in PATH (gcc recommended on Windows via MSYS2/MinGW).
rem    - OpenMP flag is -fopenmp (gcc; clang requires libomp available).
rem ============================================================================

set "CC=%CC%"
if "%CC%"=="" set "CC=gcc"

rem Recommended environment alignment
if "%CC%"=="gcc" set "PATH=C:\msys64\mingw64\bin;%PATH%"

set "CFLAGS=-O2 -Wall -Wextra -Wpedantic -g"
set "OMPFLAGS=-fopenmp"
set "BIN=bin"

echo Compiler: %CC%
echo Flags:    %CFLAGS% %OMPFLAGS%
echo Output:   %BIN%\
echo.

for %%F in (
  "src\01_basics\omp_data_sharing.c"
  "src\01_basics\omp_hello.c"
  "src\01_basics\omp_parallel_for.c"
  "src\01_basics\omp_reduction_fp.c"
  "src\01_basics\omp_reduction.c"

  "src\02_synchronization\omp_atomic_vs_critical.c"
  "src\02_synchronization\omp_barrier.c"
  "src\02_synchronization\omp_printf_interleaving.c"

  "src\03_scheduling\omp_schedule_demo_chunks.c"
  "src\03_scheduling\omp_schedule_demo.c"
  "src\03_scheduling\omp_schedule_profile_used_threads.c"
  "src\03_scheduling\omp_schedule_profile.c"

  "src\04_performance\omp_false_sharing_array.c"
  "src\04_performance\omp_false_sharing.c"
  "src\04_performance\omp_timing_reduce_max.c"
  "src\04_performance\omp_timing.c"

  "src\05_advanced\omp_reduction_fp_compensated.c"
  "src\05_advanced\omp_reduction_fp_pairwise.c"
  "src\05_advanced\omp_simd_intro.c"
  "src\05_advanced\omp_tasks_depend.c"
  "src\05_advanced\omp_tasks_intro.c"
  "src\05_advanced\omp_tasks_pipeline_gantt.c"
  "src\05_advanced\omp_tasks_pipeline_overlap.c"
) do (
  set "SRC=%%~F"
  set "REL=!SRC:src\=!"
  set "OUT=%BIN%\!REL:.c=.exe!"

  for %%D in ("!OUT!") do set "OUTDIR=%%~dpD"
  if not exist "!OUTDIR!" mkdir "!OUTDIR!" >nul 2>&1

  echo [CC] !SRC!  ^>  !OUT!
  %CC% %CFLAGS% %OMPFLAGS% "!SRC!" -o "!OUT!"
  if errorlevel 1 (
    echo [ERR] Failed: !SRC!
    exit /b 1
  )
)

echo.
echo Done.
exit /b 0
