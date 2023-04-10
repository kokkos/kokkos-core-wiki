``Kokkos::device_id``
=====================

.. role:: cppkokkos(code)
    :language: cppkokkos

Defined in header ``<Kokkos_Core.hpp>``

.. code-block:: cpp

    [[nodiscard]] int device_id() noexcept;

Returns the device id of the device that is used by the Kokkos default device
backend or ``-1`` if only host backends are enabled.

----

**See also**

.. _num_threads : num_threads.html

.. |num_threads| replace:: ``num_threads``

.. _initialize: ../initialize_finalize/initialize.html

.. |initialize| replace:: ``initialize``

.. _InitializationSettings: ../initialize_finalize/InitializationSettings.html

.. |InitializationSettings| replace:: ``InitializationSettings``

|num_threads|_: returns the number of threads used by Kokkos

|initialize|_: initializes the Kokkos execution environment

|InitializationSettings|_: settings for initializing Kokkos
