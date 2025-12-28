## Cache Coherence, Invalidation, and False Sharing

### 1. Cache Coherence in Shared-Memory Multiprocessors

Modern multicore processors implement **private per-core caches** (typically L1 and L2) backed by a shared last-level cache (LLC). To preserve the illusion of a **single coherent shared memory**, hardware implements a **cache coherence protocol**, most commonly variants of **MESI / MESIF / MOESI**.

At a conceptual level, these protocols ensure the following invariant:

> At any time, for a given memory location, all processors observe a consistent value that respects the program’s memory ordering rules.

This is achieved by associating **coherence states** with each cache line (typically 64 bytes) and by exchanging **coherence messages** (invalidate, read-for-ownership, writeback, etc.) over an interconnect.

Crucially:

* **Coherence operates at cache-line granularity**, not at variable granularity.
* Any write to *any byte* within a cache line affects ownership of the *entire* cache line.

---

### 2. Cache Line Invalidation on Writes

When a core performs a write to a memory location:

1. The core must obtain **exclusive ownership** of the cache line containing that location.
2. If other cores currently hold the same cache line in a shared or exclusive state, those copies must be **invalidated**.
3. Subsequent accesses by other cores require re-fetching the cache line, incurring latency.

This mechanism is correct and unavoidable for **true sharing**, where multiple threads intentionally read and write the same data.

However, it becomes pathological in the presence of **false sharing**.

---

### 3. Definition of False Sharing

**False sharing** occurs when:

* Multiple threads write to **distinct, logically independent variables**,
* but those variables reside on the **same cache line**.

Although there is **no logical data dependency**, the coherence protocol treats the situation as if the threads were sharing data.

Formally:

> False sharing is a performance degradation caused by cache-coherence invalidations triggered by independent writes to different variables located in the same cache line.

---

### 4. Why False Sharing Is a Hardware-Level Pathology

False sharing is fundamentally a **hardware artifact**, not a programming error in the semantic sense.

Key characteristics:

* The program is **data-race free** and **memory-model correct**.
* Synchronization constructs (`atomic`, `critical`, barriers) are not involved.
* The slowdown arises purely from **microarchitectural behavior**.

From the hardware perspective:

* Each write forces **exclusive cache-line ownership**.
* Ownership repeatedly migrates between cores.
* Cache lines “ping-pong” across cores at high frequency.

This leads to:

* Excessive coherence traffic
* Pipeline stalls
* Reduced memory-level parallelism
* Dramatically lower throughput

Importantly, **the compiler cannot generally eliminate false sharing**, because:

* It lacks semantic knowledge of per-thread access patterns.
* Layout decisions may be constrained by ABI, data structures, or external interfaces.

---

### 5. Why False Sharing Is Hard to Detect

False sharing is particularly insidious because:

* Correctness is unaffected.
* Performance degradation can be **non-linear** and highly workload-dependent.
* The problem may appear or disappear depending on:

  * Thread count
  * Scheduling
  * Cache line size
  * CPU topology
  * Alignment changes caused by small code modifications

As a result, false sharing is often discovered only through:

* Performance profiling
* Hardware performance counters (e.g., cache invalidation events)
* Controlled microbenchmarks

---

### 6. Typical False Sharing Patterns

Common sources include:

* Arrays of per-thread counters
* Structs with frequently updated fields accessed by different threads
* Adjacent elements updated by neighboring threads in block decompositions
* Padding-free thread-local data stored in shared arrays

These patterns are especially prevalent in:

* OpenMP loop parallelism
* Thread-local statistics
* Producer–consumer pipelines
* Reduction-like manual implementations

---

### 7. Mitigation Strategies

Because false sharing is a **layout problem**, mitigation focuses on **data placement**, not synchronization.

Typical strategies:

1. **Padding and alignment**

   * Place frequently written per-thread data in separate cache lines.
2. **Data privatization**

   * Use thread-private variables or OpenMP reductions.
3. **Structure transformation**

   * Convert array-of-structs (AoS) to struct-of-arrays (SoA) where appropriate.
4. **Blocking and ownership**

   * Ensure that threads operate on disjoint cache-line regions.
5. **Explicit alignment**

   * Use language or compiler alignment directives when available.

Each strategy trades memory footprint or code complexity for performance stability.

---

### 8. Relationship to OpenMP

OpenMP provides **correctness guarantees**, not cache-efficiency guarantees.

Key observations:

* OpenMP does not prevent false sharing.
* `atomic` and `critical` do not solve false sharing (they may worsen it).
* Reductions are often preferable because they:

  * Use private per-thread storage
  * Minimize coherence traffic
  * Defer aggregation to a low-frequency synchronization phase

Therefore, understanding false sharing is essential for writing **scalable OpenMP programs**.

---

### 9. Summary

False sharing is a **hardware-induced performance pathology** arising from the interaction of:

* Cache-line granularity
* Coherence protocols
* Independent per-thread writes

It is invisible at the language and memory-model level, yet can dominate runtime behavior.

A programmer who understands false sharing reasons not only about **what data is shared**, but also **how data is laid out in memory**—an essential skill in high-performance shared-memory programming.

---

### Suggested References

* OpenMP Architecture Review Board, *OpenMP API Specification*
* Intel® 64 and IA-32 Architectures Optimization Reference Manual
* ARM® Architecture Reference Manual (cache and coherence chapters)
* McKenney, *Is Parallel Programming Hard, And, If So, What Can You Do About It?*
* Hennessy & Patterson, *Computer Architecture: A Quantitative Approach*
