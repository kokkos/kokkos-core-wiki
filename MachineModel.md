# Chapter 2

# Machine Model

After reading this chapter you will understand the abstract model of a parallel computing node which underlies the design choices and structure of the Kokkos framework. The machine model ensures the applications written using Kokkos will have portability across architectures while being performant on a range of hardware.

The machine model has two important components:
* _Memory spaces_, in which data structures can be allocated
* _Execution spaces_, which execute parallel operations using data from one or more _memory spaces_.

## 2.1 Motivations

Kokkos is comprised of two orthogonal aspects. The first of these is an underlying
_abstract machine model_ which describes fundamental concepts required for the development of future portable and performant high performance computing applications; the second is a _concrete instantiation of the programming model_ written in C++, which allows programmers to write to the concept machine model. It is important to treat these two aspects of Kokkos as distinct entities because the underlying model being used by Kokkos could, in the future, be instantiated in additional languages beyond C++ yet the algorithmic specification would remain valid.

### 2.1.1 Kokkos Abstract Machine Model
Kokkos assumes an _abstract machine model_ for the design of future shared-memory computing architectures. The model (shown in Figure 2.1) assumes that there may be multiple execution units in a compute node. For a more general discussion of abstract machine models for Exascale computing the reader should consult reference Ang<sup>1</sup>. In the figure shown here, we have elected to show two different types of compute units - one which represents multiple latency-optimized cores, similar to contemporary processor cores, and a second source of compute in the form of an off die accelerator. Of note is that the processor and accelerator each have distinct memories, each with unique performance properties, that may or may not be accessible across the node (i.e. the memory may be reachable or _shared_ by all execution units, but specific memory spaces may also be only accessible by specific execution units). The specific layout shown in Figure 2.1 is an instantiation of the Kokkos abstract machine model used to describe the potential for multiple types of compute engines and memories within a single node. In future systems, there may be a range of execution engines which are used in the node ranging from a single type of core, as in many/multi-core processors found today, through to a range of execution units where many-core processors may be joined to numerous types of accelerator cores. In order to ensure portability to the potential range of nodes, an abstraction of the compute engines and available memories are required. 
***
<sup>1</sup> Ang, J.A., et. al., **Abstract Machine Models and Proxy Architectures for Exascale Computing**,
2014, Sandia National Laboratories and Lawrence Berkeley National Laboratory, DOE Computer Architecture Laboratories Project
***

![node](https://github.com/kokkos/ProgrammingGuide/blob/figure-edits/figures/kokkos-node-doc.png)

<h4>Figure 2.1 Conceptual Model of a Future High Performance Computing Node</h4>

## 2.2 Kokkos Execution Spaces
Kokkos uses the term _execution spaces_ to describe a logical grouping of computation units which share an identical set of performance properties. An execution space provides a set of parallel execution resources which can be utilized by the programmer using several types of fundamental parallel operation. For a list of the operations available see Chapter 7.

### 2.2.1 Execution Space Instances
An _instance_ of an execution space is a specific instantiation of an execution space to which a programmer can target parallel work. By means of example, an execution space might be used to describe a multi-core processor. In this example, the execution space contains several homogeneous cores which share some logical grouping. In a program written to the Kokkos model, an instance of this execution space would be made available on which parallel kernels could be executed. As a second example, if we were to add a GPU to the multi-core processor so a second execution space type is available in the system, the application programmer would then have two execution space instances available to select from. The important consideration here is that the method of compiling code for different execution spaces and the dispatch of kernels to instances is abstracted by the Kokkos model. This allows application programmers to be free from writing algorithms in hardware specific languages.

![execution-space](https://github.com/kokkos/ProgrammingGuide/blob/figure-edits/figures/kokkos-execution-space-doc.png)

<h4>Figure 2.2 Example Execution Spaces in a Future Computing Node</h4>

### 2.2.2 Kokkos Memory Spaces
The multiple types of memory which will become available in future computing nodes are abstracted by Kokkos through _memory spaces_. Each memory space provides a finite storage capacity at which data structures can be allocated and accessed. Different memory space types have different characteristics with respect to accessibility from execution spaces as well as their performance characteristics.

### 2.2.3 Instances of Kokkos Memory Spaces
In much the same way execution spaces have specific instantiations through the availability of an _instance_ so do memory spaces. An instance of a memory space provides a concrete method for the application programmer to request data storage allocations. Returning to the examples provided for execution spaces, the multi-core processor may have multiple memory spaces available including on-package memory, slower DRAM and additional sets of non-volatile memories. The GPU may also provide an additional memory space through its local on-package memory. The programmer is free to decide where each data structure may be allocated by requesting these from the specific instance associated with that memory space. Kokkos provides the appropriate abstraction of the allocation routines and any associated data management operations including releasing the memory, returning it for future use, as well as for copy operations.

![memory-space](https://github.com/kokkos/ProgrammingGuide/blob/figure-edits/figures/kokkos-memory-space-doc.png)

<h4>Figure 2.3 Example Memory Spaces in a Future Computing Node</h4>

**Atomic accesses to Memory in Kokkos** In cases where multiple executing threads attempt to read a memory address, complete a computation on the item, and write it back to same address in memory, an ordering collision may occur. These situations, known as _race conditions_ (because the data value stored in memory as the threads complete is dependent on which thread completes its memory operation last), are often the cause of non-determinism in parallel programs. A number of methods can be employed to ensure that race conditions do not occur in parallel programs including the use of locks (which allow only a single thread to gain access to data structure at a time), critical regions (which allow only one thread to execute a code sequence at any point in time) and _atomic_ operations. Memory operations which are atomic guarantee that a read, simple computation, and write to memory are completed as a single unit. This might allow application programmers to safely increment a memory value for instance, or more commonly, to safely accumulate values from multiple threads into a single memory location.

**Memory Consistency in Kokkos** Memory consistency models are a complex topic in and of themselves and usually rely on complex operations associated with hardware caches or memory access coherency (for more information see Hennessy and Paterson, 2011). Kokkos does not _require_ caches to be present in hardware and so assumes an extremely weak memory consistency model. In the Kokkos model, the programmer should not assume any specific ordering of memory operations being issued by a kernel. This has the potential to create race conditions between memory operations if these are not appropriately protected. In order to provide a guarantee that memory operations are completed, Kokkos provides a _fence_ operation which forces the compute engine to complete all outstanding memory operations before any new ones can be issued. With appropriate use of fences, programmers are thereby able to ensure that guarantees can be made as to when data will _definitely_ have been written to memory.

## 2.3 Program execution

It is tempting to try to define formally what it means for a processor to execute code. None of us authors have a background in logic or what computer scientists call "formal methods," so our attempt might not go very far! We will stick with informal definitions and rely on Kokkos' C++ implementation as an existence proof that the definitions make sense.

Kokkos lets users tell execution spaces to execute parallel operations. These include parallel for, reduce, and scan (see Chapter 7) as well as View allocation and initialization (see Chapters 6 and 5). We name the class of all such operations _parallel dispatch_.

From our perspective, there are three kinds of code:

1. Code executing inside of a Kokkos parallel operation
1. Code outside of a Kokkos parallel operation that asks Kokkos to do something (e.g., parallel dispatch itself)
1. Code that has nothing to do with Kokkos

The first category is the most restrictive. Section 8.2 explains restrictions on inter-team synchronization. In general, we limit the ability of Kokkos-parallel code to invoke Kokkos operations (other than for nested parallelism; see Chapter 8 and especially Section 8.2). We also forbid dynamic memory allocation (other than from the team's scratch pad) in parallel operations. Whether Kokkos-parallel code may invoke operating system routines or third-party libraries depends on the execution and memory spaces being used. Regardless, restrictions on inter-team synchronization have implications for things like filesystem access.

_Kokkos threads are for computing in parallel_, not for overlapping I/O and computation, and not for making graphical user interfaces responsive. Use other kinds of threads (e.g., operating system threads) for the latter two purposes. You may be able to mix Kokkos' parallelism with other kinds of threads; see Section 2.3.1. Kokkos' developers are also working on a task parallelism model that will work with Kokkos' existing data-parallel constructs. 

**Reproducible reductions and scans** Kokkos promises _nothing_ about the order in which the iterations of a parallel loop occur. However, it _does_ promise that if you execute the same parallel reduction or scan, using the same hardware resources and run-time settings, then you will get the same results each time you run the operation. "Same results" even means "with respect to floating-point rounding error."

**Asynchronous parallel dispatch** This concerns the second category of code that calls Kokkos operations. In Kokkos, parallel dispatch executes _asynchronously_. This means that it may return "early," before it has actually completed. Nevertheless, it executes _in sequence_ with respect to other Kokkos operations on the same execution or memory space. This matters for things like timing. For example, a `parallel_for` may return "right away," so if you want to measure how long it takes, you must first call `fence()` on that execution space. This forces all functors to complete before `fence()` returns.

### 2.3.1 Thread safety?

Users may wonder about "thread safety," that is, whether multiple operating system threads may safely call into Kokkos concurrently. Kokkos' thread safety depends on both its implementation and on the execution and memory spaces that the implementation uses. The C++ implementation has made great progress towards (non-Kokkos) thread safety of View memory management. For now, however, the most portable approach is for only one (non-Kokkos) thread of execution to control Kokkos. Also, be aware that operating system threads might interfere with Kokkos' performance depending on the execution space that you use.
