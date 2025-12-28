# CPU Cores and Logical Processors

## Abstract

CPU cores and logical processors are fundamental abstractions used to describe parallel execution capabilities in modern processors. This paper explains the distinction between physical cores and logical processors as exposed by contemporary CPU architectures and operating systems. The scope includes architectural concepts, hardware multithreading mechanisms, and scheduling implications. The main conclusion is that logical processors increase hardware utilization but do not provide linear performance scaling relative to physical cores.

---

## Introduction

### Context

* Modern CPUs execute multiple instruction streams concurrently.
* Parallelism is achieved through **multiple cores** and **hardware multithreading**.
* Operating systems schedule work based on visible processing units.

### Motivation

* System monitors expose both “cores” and “logical processors.”
* These terms are often misunderstood or treated as equivalent.
* Incorrect interpretation leads to flawed performance expectations.

### Objective

* Precisely define CPU cores and logical processors.
* Systematically distinguish architectural and logical views.
* Explain performance and scheduling implications.

---

## Core Concepts and Definitions

### Fundamental Terms

**Table 1 – Core Definitions**

| Term                              | Definition                                                                |
| --------------------------------- | ------------------------------------------------------------------------- |
| CPU core                          | A physical execution unit capable of independently executing instructions |
| Logical processor                 | A hardware-visible execution context scheduled by the operating system    |
| Hardware thread                   | Architectural state allowing one instruction stream to execute on a core  |
| Simultaneous Multithreading (SMT) | Technique allowing multiple hardware threads per core                     |
| Hyper-Threading                   | Intel’s implementation of SMT                                             |
| OS scheduler                      | Kernel component assigning threads to logical processors                  |

---

## Systematic Classification / Breakdown

### Processing Unit Types

**Table 2 – Classification of Processing Units**

| Category          | Definition                           | Purpose                   | Key Characteristics                      |
| ----------------- | ------------------------------------ | ------------------------- | ---------------------------------------- |
| Physical core     | Independent silicon execution engine | True parallel execution   | Own pipeline, registers, execution units |
| Logical processor | Hardware thread exposed to OS        | Improve utilization       | Shares core resources                    |
| Software thread   | OS-managed execution unit            | Program-level concurrency | Mapped to logical processors             |

---

### Physical CPU Core

* **Definition**: Independent hardware unit with full instruction pipeline.
* **Purpose**: Execute one instruction stream without sharing execution resources.
* **Key characteristics**:

  * Dedicated ALUs and FPUs
  * Private architectural state
  * Limited by silicon area and power

---

### Logical Processor

* **Definition**: Hardware-visible execution context within a core.
* **Purpose**: Increase instruction throughput by hiding stalls.
* **Key characteristics**:

  * Shares execution units with sibling thread
  * Own register set
  * Not equivalent to a full core

---

## Technical Analysis

### Hardware Multithreading Mechanism

**Step-by-step SMT operation**

* Core maintains multiple architectural states.
* Instruction fetch alternates or interleaves between states.
* Execution units are dynamically shared.
* Stalled thread yields execution slots to another.

---

### Resource Sharing Model

**Table 3 – Resource Allocation in SMT Cores**

| Resource             | Shared  | Private |
| -------------------- | ------- | ------- |
| Register file        | No      | Yes     |
| Execution units      | Yes     | No      |
| L1 cache             | Usually | No      |
| Instruction pipeline | Yes     | No      |
| Program counter      | No      | Yes     |

---

### Scheduling Model

**OS-level workflow**

* OS enumerates logical processors.
* Scheduler assigns software threads to logical processors.
* Core-level arbitration resolves resource contention.

---

## Comparative Overview

### Core vs Logical Processor

**Table 4 – Comparative Analysis**

| Aspect              | Physical Core    | Logical Processor         |
| ------------------- | ---------------- | ------------------------- |
| Hardware reality    | Physical silicon | Architectural abstraction |
| Parallelism         | True             | Opportunistic             |
| Resource ownership  | Dedicated        | Shared                    |
| Performance scaling | Near-linear      | Sub-linear                |
| Power cost          | High             | Low                       |

---

### Performance Characteristics

**Table 5 – Performance Comparison**

| Metric                  | Single Core | Core with SMT      |
| ----------------------- | ----------- | ------------------ |
| Maximum throughput      | Baseline    | +10–30% typical    |
| Latency per thread      | Low         | Potentially higher |
| Determinism             | High        | Reduced            |
| Worst-case interference | None        | Present            |

---

## Practical Implications

### Facts

* Operating systems schedule on logical processors.
* SMT improves average throughput.
* Workloads with stalls benefit most from SMT.
* Compute-bound workloads show limited gains.

### Assumptions

* Logical processors are not equivalent to cores.
* SMT effectiveness depends on workload behavior.
* Real-time systems may disable SMT for predictability.

---

## Conclusion

* CPU cores are physical execution units.
* Logical processors are hardware threads exposed to software.
* SMT increases utilization but shares core resources.
* Performance gains are workload-dependent.
* Accurate distinction is essential for performance analysis and system design.

---

### Summary Table

**Table 6 – Summary of Key Distinctions**

| Concept               | CPU Core   | Logical Processor |
| --------------------- | ---------- | ----------------- |
| Physical entity       | Yes        | No                |
| Independent execution | Yes        | No                |
| OS-visible            | Indirectly | Yes               |
| Resource sharing      | None       | Extensive         |
