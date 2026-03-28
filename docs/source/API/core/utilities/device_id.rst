``Kokkos::device_id``
=====================

.. role:: cpp(code)
    :language: cpp

Defined in header ``<Kokkos_Core.hpp>``

.. code-block:: cpp

    [[nodiscard]] int device_id() noexcept;  // (since 4.1)

Returns the id of the device that is used by ``DefaultExecutionSpace`` or
``-1`` if only host backends are enabled.

----

**See also**

:doc:`num_devices <num_devices>`: returns the number of devices available to Kokkos

:doc:`num_threads <num_threads>`: returns the number of threads used by Kokkos

:doc:`initialize <../initialize_finalize/initialize>`: initializes the Kokkos execution environment

:doc:`InitializationSettings <../initialize_finalize/InitializationSettings>`: settings for initializing Kokkos
