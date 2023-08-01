``Kokkos::printf``
==================

.. role:: cppkokkos(code)
    :language: cppkokkos

Defined in header ``<Kokkos_Core.hpp>``

.. code-block:: cpp

    template <typename... Args>
    KOKKOS_FUNCTION void printf(const char* format, Args... args);

Prints the data specified in ``format`` and ``args...`` to ``stdout``.
The behavior is analogous to ``std::printf``, but the return type is ``void``
to ensure a consistent behavior across backends.
