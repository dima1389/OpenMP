### omp_tasks_depend.c

* **Level**: Advanced (task graphs and dependencies)

* **Complements**:

  * `omp_tasks_intro.c` (basic task creation)
  * `omp_schedule_demo*.c` (loop scheduling)

* **Teaches explicitly**:

  * How to encode ordering constraints *without* global synchronization
  * Why `depend` is superior to barriers for pipelines and DAGs
  * How OpenMP tasking scales to irregular parallel workloads
