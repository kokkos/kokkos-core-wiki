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

.. _device_id : device_id.html

.. |device_id| replace:: ``device_id``

.. _num_devices : num_devices.html

.. |num_devices| replace:: ``num_devices``

.. _initialize: ../initialize_finalize/initialize.html

.. |initialize| replace:: ``initialize``

.. _InitializationSettings: ../initialize_finalize/InitializationSettings.html

.. |InitializationSettings| replace:: ``InitializationSettings``

|num_devices|_: returns the number of devices available to Kokkos

|device_id|_: returns the id of the device used by Kokkos

|initialize|_: initializes the Kokkos execution environment

|InitializationSettings|_: settings for initializing Kokkos
