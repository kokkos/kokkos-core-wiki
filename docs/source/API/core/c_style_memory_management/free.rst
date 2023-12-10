``kokkos_free``
===============

.. role:: cppkokkos(code)
    :language: cppkokkos

Defined in header ``<Kokkos_Core.hpp>``

.. _Kokkos_kokkos_malloc: ./malloc.html

.. |Kokkos_kokkos_malloc| replace:: ``Kokkos::kokkos_malloc()``

.. _Kokkos_kokkos_realloc: ./realloc.html

.. |Kokkos_kokkos_realloc| replace:: ``Kokkos::kokkos_realloc()``

Deallocates the space previously allocated by |Kokkos_kokkos_malloc|_ or |Kokkos_kokkos_realloc|_ on the specified memory space ``MemorySpace``.

If ``ptr`` is a null pointer, the function does nothing.

Description
-----------

.. cppkokkos:function:: template <class MemorySpace = Kokkos::DefaultExecutionSpace::memory_space> void kokkos_free(void* ptr);

    :tparam MemorySpace: Controls the storage location. If omitted the memory space of the default execution space is used (i.e. ``Kokkos::DefaultExecutionSpace::memory_space``).

    :param ptr: The pointer to the memory to deallocate on the specified memory space.

    :returns: (none)

    :throws: Throws ``std::runtime_error`` on failure to deallocate.
