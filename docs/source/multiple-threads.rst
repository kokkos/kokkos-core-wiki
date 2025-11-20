Use from Multiple Threads
#########################

This document describes how various Kokkos functionality can safely be called when using more than one thread of host execution.
It also describes intentions for what Kokkos aspires to implement in support of known or anticipated use cases, with clear indications that they are not currently supported.
Some discussion of implementation characteristics and performance expectations is included as well.

We will use terms as defined by the C++ standard memory model. In that model, executions that include a data race have undefined behavior.
Thus, use of Kokkos in ways that don't conform to the safety guidance given here may be considered to result in undefined behavior.

In formal terms, we address two distinct issues here.
*Concurrency* is the matter of what operations may safely execute in different threads. 
*Linearizability* is the matter of how the results of concurrently-executed operations may map to potential observed orders in a hypothetical non-concurrent execution.

Core
====

Fundamental Operations
----------------------

For purposes of multi-threaded execution, all free and member functions provided by Kokkos can be thought of as performing a sequence of one or more of the following *fundamental operations*


* *Initialization*

.. Not just Kokkos::init, but also whatever device-specific or thread-specific stuff we have Legion doing now

* *Finalization*

.. Ditto Initialization

* *Memory Allocation*


* *Memory Deallocation*


* *Parallel Dispatch*
  ``parallel_for``, ``parallel_reduce``, ``parallel_scan``, each of which occurs on a particular *Execution Space Instance*. If none is specified, Kokkos internally uses a default *Execution Space Instance*.

* *Data Movement*
  ``deep_copy`` which occurs on a particular *Execution Space Instance*. 
.. The 'two-argument' overload of the ``deep_copy()`` function performs a *Global Synchronization*, a ``deep_copy`` operation on an internal *Execution Space Instance*, and another *Global Synchronization* operation.

* *Global Synchronization*

..  Like what ``Kokkos::fence()`` does

* *Local Synchronization*

.. Like what we get from  ``execution_space_instance.fence()``, hypothetical ``is_running`` that returns ``false``

* *Data Access*
  ``View::operator()``, to memory that is accessible from the host.



Fundamental Object Instances
----------------------------

The fundamental objects of Kokkos programming, such as ``View`` and instances of execution space types, have general semantics similar to a reference-counted pointer.
These objects provide no surface-level concurrency control of operations like assignment. Thus, client code is responsible for serializing any such operations between threads, such that they establish a *happens-before* relationship between writes and any other operations (e.g. by locking).
The descriptions that follow refer to operations performed on the underlying object that is pointed-to by the C++ object that client code uses.

Basic *Concurrency*
-------------------

A program that is single-threaded outside its use of any multi-threaded Kokkos backend can execute code that performs any otherwise-valid sequence of *Fundamental Operations*. (Note: This is analogous to ``MPI_THREAD_SINGLE``)

A program that performs *Initialization* in a particular thread, and only executes code that performs any subsequent *Fundamental Operations* on that thread, is equivalent to a single-threaded program for these purposes. (Note: This is analogous to ``MPI_THREAD_FUNNELED``)

A multi-threaded program structured such that there is a *happens-before* relationship between each call to perform a *Fundamental Operation* will behave equivalently to a single-threaded program that performs the same sequence of *Fundamental Operations*. (Note: This is analogous to ``MPI_THREAD_SERIALIZED``)

.. Do we actually want to guarantee that every Fundamental Operation is serializing? Should that just mean that we don't require call sites to have *happens-before* relationships, or should they also internally create such *happens-before* relationships? I.e. that the calling threads *synchronize-with* each other at those points?


Execution Space Instance
------------------------

An *Execution Space Instance* denotes a thread-like entity on which *Fundamental Operations* can be performed. Those *Fundamental Operations* have a chain of *sequenced-before* relationships corresponding to the order in which host code performs them.

.. Assuming we guarantee internal serialization, the following would apply
.. - If distinct host threads perform *Fundamental Operations* on a common *Execution Space Instance* without a *happens-before* relationship between the calls, then their sequence in such a chain is indeterminate.
.. Otherwise, it would be undefined behavior

*Local Synchronization* creates a *happens-before* relationship between the completion of every *Fundamental Operation* on the specified *Execution Space Instance* that *happens-before* the *Local Synchronization* and the thread that performs the *Local Synchronization*.

*Global Synchronization* creates a *happens-before* relationship between the completion of every *Fundamental Operation* on any *Execution Space Instance* that *happens-before* the *Global Synchronization* and the thread that performs the *Global Synchronization*.

.. Should the above actually be *synchronizes-with*?

A multi-threaded program may concurrently perform *Fundamental Operations* on distinct *Execution Space Instances*. The order in which these operations execute is indeterminate.

``View``
--------

* Managed Construction
  Managed construction of a Kokkos View performs a *Memory Allocation*, potentially followed by a *Parallel Dispatch* to initialize the memory (depending on whether ``WithoutInitializing`` was passed), potentially followed by a *Synchronization* (if no execution space instance was passed, so that allocation and initialization *happen-before* any subsequent operation that may reference the ``View``'s memory').
  .. Do we want that to be *Global Synchronization* or *Local Synchronization*?
  (Note that particular backends may internally perform additional synchronization operations)
* Unmanaged Construction
  Unmanaged construction performs no fundamental operations.
* Destruction
  Destruction occurs when the reference count of the underlying object to which a ``View`` refers falls to zero. Destruction performs a *Global Synchronization*, followed by a *Memory Deallocation*.
  The *Global Synchronization* is performed to ensure that any preceding operations that may reference the memory owned by the underlying object *happen-before* that memory is released.
  Any *Fundamental Operation* which accesses the ``View`` instance performed in a way that it does not have a *happens-before* relationship with Destruction results in undefined behavior.
  (Note this could occur via improper concurrent use of a ``View`` variable shared between threads, one of which passes it to an operation, and another which reassigns it to reference a different instance)
* Metadata Query
* Element Access
  Element Access performs a Data Access operation.


Backend-Specific Details
------------------------

.. Local or Global synchronizations below?

* ``Serial``
  This backend is *synchronous* - there is implicitly a *Synchronization* after any *Memory Allocation*, *Memory Deallocation*, *Parallel Dispatch*, or *Data Movement* operation. 

* ``OpenMP``
  This backend is *synchronous* - there is implicitly a *Synchronization* after any *Memory Allocation*, *Memory Deallocation*, *Parallel Dispatch*, or *Data Movement* operation. 

* ``Threads``
  This backend is *synchronous* - there is implicitly a *Synchronization* after any *Memory Allocation*, *Memory Deallocation*, *Parallel Dispatch*, or *Data Movement* operation. 

* ``CUDA`` and ``HIP``

* ``HPX``


Performing Fundamental Operations on Device Execution Space Instances Within Operations on Host Execution Space Instances
------------------------------------------------------------------------------------------------------------------------------------------

.. ???


Containers
==========




Algorithms
==========



Kokkos Kernels
==============