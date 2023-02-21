``kokkos_malloc``
=================

.. role:: cppkokkos(code)
    :language: cppkokkos

Defined in header ``<Kokkos_Core.hpp>``

.. code-block:: cpp

    template <class MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
    void* kokkos_malloc(const std:string& label, size_t size);

.. code-block:: cpp

    template <class MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
    void* kokkos_malloc(size_t size);

.. _MemorySpace: ../memory_spaces.html

.. |MemorySpace| replace:: ``MemorySpace``

Allocate ``size`` bytes of uninitialized storage on the specified memory space |MemorySpace|_ plus some extra space for metadata such as the label.

If allocation succeeds, returns a pointer to the lowest (first) byte in the allocated memory block that is suitably aligned for any scalar type.

If allocation fails, an exception of type ``Kokkos::Experimental::RawMemoryAllocationFailure`` is thrown.

WARNING: calling any function that manipulates the behavior of the memory (e.g. ``memAdvise``) on memory managed by ``Kokkos`` results in undefined behavior.

Parameters
----------

``label``: A user provided string which is used in profiling and debugging tools via the KokkosP Profiling Tools.  
``size``: The number of bytes to allocate.

Template parameters
-------------------

* ``MemorySpace``: Controls the storage location. If omitted the memory space of the default execution space is used (i.e. ``Kokkos::DefaultExecutionSpace::memory_space``).

Return value
------------

.. _Kokkos_kokkos_free: free.html

.. |Kokkos_kokkos_free| replace:: ``Kokkos::kokkos_free()``

.. _Kokkos_realloc: realloc.html

.. |Kokkos_realloc| replace:: ``Kokkos::realloc()``

On success, returns the pointer to the beginning of newly allocated memory.
To avoid a memory leak, the returned pointer must be deallocated with |Kokkos_kokkos_free|_ or |Kokkos_realloc|_.

Exceptions
----------

On failure, throws ``Kokkos::Experimental::RawMemoryAllocationFailure``.
