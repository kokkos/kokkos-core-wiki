``Kokkos::printf``
==================

.. role:: cppkokkos(code)
    :language: cppkokkos

Defined in header ``<Kokkos_Core.hpp>``

.. code-block:: cpp

    template <typename... Args>
    KOKKOS_FUNCTION void printf(const char* format, Args... args);  // (since 4.2)

Prints the data specified in ``format`` and ``args...`` to ``stdout``.
The behavior is analogous to ``std::printf``, but the return type is ``void``
to ensure a consistent behavior across backends.

Example
~~~~~~~

.. code-block:: cpp

    #include <Kokkos_Core.hpp>

    int main(int argc, char* argv[]) {
        Kokkos::initialize(argc, argv);
        Kokkos::parallel_for(4, KOKKOS_LAMBDA(int i) {
            Kokkos::printf("hello world from thread %d\n", i);
        });
        Kokkos::finalize();
    }

Notes
~~~~~
* The ``Kokkos::printf()`` function was added in release 4.2
* Calling ``Kokkos::printf()`` from a kernel may affect register usage and affect performance.
