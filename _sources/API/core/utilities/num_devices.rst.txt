``Kokkos::num_devices``
=======================

.. role:: cppkokkos(code)
    :language: cppkokkos

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

.. _device_id : device_id.html

.. |device_id| replace:: ``device_id``

.. _num_threads : num_threads.html

.. |num_threads| replace:: ``num_threads``

.. _initialize: ../initialize_finalize/initialize.html

.. |initialize| replace:: ``initialize``

.. _InitializationSettings: ../initialize_finalize/InitializationSettings.html

.. |InitializationSettings| replace:: ``InitializationSettings``

|device_id|_: returns the id of the device used by Kokkos

|num_threads|_: returns the number of threads used by Kokkos

|initialize|_: initializes the Kokkos execution environment

|InitializationSettings|_: settings for initializing Kokkos

