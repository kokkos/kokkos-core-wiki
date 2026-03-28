``Kokkos::num_threads``
=======================

.. role:: cpp(code)
    :language: cpp

Defined in header ``<Kokkos_Core.hpp>``

.. code-block:: cpp

    [[nodiscard]] int num_threads() noexcept;  // (since 4.1)

Returns the number of concurrent threads that are used by ``DefaultHostExecutionSpace``.

----

**See also**

:doc:`num_devices <num_devices>`: returns the number of devices available to Kokkos

:doc:`device_id <device_id>`: returns the id of the device used by Kokkos

:doc:`initialize <../initialize_finalize/initialize>`: initializes the Kokkos execution environment

:doc:`InitializationSettings <../initialize_finalize/InitializationSettings>`: settings for initializing Kokkos
