``Kokkos::num_threads``
======================

.. role:: cppkokkos(code)
    :language: cppkokkos

Defined in header ``<Kokkos_Core.hpp>``

.. code-block:: cpp

    [[nodiscard]] int num_threads() noexcept;

Returns the number of concurrent threads that are used by ``DefaultHostExecutionSpace``.

----

**See also**

.. _device_id : device_id.html

.. |device_id| replace:: ``device_id``

.. _initialize: ../initialize_finalize/initialize.html

.. |initialize| replace:: ``initialize``

.. _InitializationSettings: ../initialize_finalize/InitializationSettings.html

.. |InitializationSettings| replace:: ``InitializationSettings``

|device_id|_: returns the id of the device used by Kokkos

|initialize|_: initializes the Kokkos execution environment

|InitializationSettings|_: settings for initializing Kokkos
