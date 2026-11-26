``kokkos_free``
===============

.. role:: cpp(code)
    :language: cpp

Defined in header ``<Kokkos_Core.hpp>``

Deallocates the space previously allocated by :cpp:func:`kokkos_malloc` or :cpp:func:`kokkos_realloc` on the specified memory space ``MemorySpace``.

If ``ptr`` is a null pointer, the function does nothing.

Description
-----------

.. cpp:function:: template <class MemorySpace = Kokkos::DefaultExecutionSpace::memory_space> void kokkos_free(void* ptr);

    :tparam MemorySpace: Controls the storage location. If omitted the memory space of the default execution space is used (i.e. ``Kokkos::DefaultExecutionSpace::memory_space``).

    :param ptr: The pointer to the memory to deallocate on the specified memory space.

    :returns: (none)

    :throws: Throws ``std::runtime_error`` on failure to deallocate.
