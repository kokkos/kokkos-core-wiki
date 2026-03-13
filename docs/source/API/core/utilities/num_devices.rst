``Kokkos::num_devices``
=======================

.. role:: cpp(code)
    :language: cpp

Defined in header ``<Kokkos_Core.hpp>``

.. code-block:: cpp

    [[nodiscard]] int num_devices() noexcept;  // (since 4.3)

Returns the number of available devices on the system or ``-1`` if only host backends are enabled.

Notes
-----

``Kokkos::num_devices()`` may be used to determine the number of devices that
are available to Kokkos for execution.
It is one of the few runtime functions that may be called before
``Kokkos::initialize()`` or after ``Kokkos::finalize()``.

Example
-------

.. code-block:: cpp

   #include <Kokkos_Core.hpp>
   #include <iostream>

   int main(int argc, char* argv[]) {
     if (Kokkos::num_devices() == 0) {
       std::cerr << "no device available for execution\n";
       return 1;
     }
     Kokkos::initialize(argc, argv);
     // do stuff
     Kokkos::finalize();
     return 0;
   }


----

**See also**

:doc:`device_id <device_id>`: returns the id of the device used by Kokkos

:doc:`num_threads <num_threads>`: returns the number of threads used by Kokkos

:doc:`initialize <../initialize_finalize/initialize>`: initializes the Kokkos execution environment

:doc:`InitializationSettings <../initialize_finalize/InitializationSettings>`: settings for initializing Kokkos
