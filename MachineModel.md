# Chapter 2

# Machine Model

After reading this chapter you will understand the abstract model of a parallel computing node which underlies the design choices and structure of the Kokkos framework. The machine model ensures the applications written using Kokkos will have portability across architectures while being performant on a range of hardware.

The machine model has two important components:
* _Memory spaces_, in which data structures can be allocated
* _Execution spaces_, which execute parallel operations using data from one or more _memory spaces_.

## 2.1 Motivations

Kokkos is comprised of two orthogonal aspects. The first of these is an underlying
_abstract machine model_ which describes fundamental concepts required for the development of future portable and performant high performance computing applications; the second is a concrete instantiation of the programming model written in C++, which allows programmers to write to the concept machine model. It is important to treat these two aspects to Kokkos as distinct entities because the underlying model being used by Kokkos could, in the future, be instantiated in additional languages beyond C++ yet the algorithmic specification would remain valid.

### 2.1.1 Kokkos Abstract Machine Model
Kokkos assumes an _abstract machine model_ for the design of future shared-memory
computing architectures. The model (shown in Figure 2.1) assumes that there may be
multiple execution units in a compute node. For a more general discussion of abstract
machine models for Exascale computing the reader should consult reference [?]. In the
figure shown here, we have elected to show two different types of compute units - one
which represents multiple latency-optimized cores, similar to contemporary processor
cores, and a second source of compute in the form of an off die accelerator. Of note
is that the processor and accelerator each have distinct memories, each with unique
performance properties, that may or may not be accessible across the node (i.e. the
memory may be reachable or _shared_ by all execution units, but specific memory spaces
may also be only accessible by specifc execution units). The specific layout shown in
Figure 2.1 is an instantiation of the Kokkos abstract machine model used to describe the
potential for multiple types of compute engines and memories within a single node. In
future systems there may be a range of execution engines which are used in the node
ranging from a single type of core as in many/multi-core processors found today through
to a range of execution units where many-core processors may be joined to numerous
types of accelerator cores. In order to ensure portability to the potential range of nodes
an abstraction of the compute engines and available memories are required.

[Insert Figure 2.1 here; 
   caption: Figure 2.1 Conceptual Model of a Future High Performance Computing Node ]

## 2.2 Kokkos Execution Spaces
Kokkos uses the term _execution spaces_ to describe a logical grouping of computation units
which share an indentical set of performance properties. An execution space provides
a set of parallel execution resources which can be utilized by the programmer using
several types of fundamental parallel operation. For a list of the operations available see
Chapter ??.

### 2.2.1 Execution Space Instances
An _instance_ of an execution space is a specific instantiation of an execution space to
which a programmer can target parallel work. By means of example, an execution space
might be used to describe a multi-core processor. In this example, the execution space
contains several homogeneous cores which share some logical grouping. In a program
written to the Kokkos model, an instance of this execution space would be made available
on which parallel kernels could be executed. As a second example, if we were to add a
GPU to the multi-core processor a second execution space type is available in the system,
the application programmer would then have two execution space instances available to
select from. The important consideration here is that the method of compiling code for
different execution spaces and the dispatch of kernels to instances is abstracted by the
Kokkos model. This allows application programmers to be free from writing algorithms
in hardware specific languages.

[Insert Figure 2.2 here; 
   caption: Figure 2.2 Example Execution Spaces in a Future Computing Node ]

### 2.2.2 Kokkos Memory Spaces
The multiple types of memory which will become available in future computing nodes are abstracted by Kokkos through memory spaces. Each memory space provides a finite storage capacity at which data structures can be allocated and accessed. Different memory space types have different characteristics with respect to accesibility from execution spaces as well as their performance characteristics.

### 2.2.3 Instances of Kokkos Memory Spaces
In much the same way execution spaces have specific instantiations through the availability of an instance so do memory spaces. An instance of a memory space provides a concrete method for the application programmer to request data storage allocations. Returning to the examples provided for execution spaces, the multi-core processor may have multiple memory spaces available including on-package memory, slower DRAM and additional a set of non-volatile memories. The GPU may also provide an additional memory space through its local on-package memory. The programmer is free to decide where each data structure may be allocated by requesting these from the specific instance associated with that memory space. Kokkos provides the appropriate abstraction of the allocation routines and any associated data management operations including releasing the memory and returning it for future use, as well as copy operations.

**Atomic accesses to Memory in Kokkos** In cases where multiple executing threads
attempt to read a memory address, complete a computation on the item and write it back
to same address in memory, an ordering collision may occur. These situations, known
as race conditions (because the data value stored in memory as the threads complete is
dependent on which thread completes its memory operation last), are often the cause
of non-determinism in parallel programs. A number of methods can be employed to
ensure that race conditions do not occur in parallel programs including the use of locks
(which allow only a single thread to gain access to data structure at a time), critical