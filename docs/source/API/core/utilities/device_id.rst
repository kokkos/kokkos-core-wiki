``device_id``
=============

Defined in header ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

    Kokkos::device_id();

Returns the id of the device that is used by ``DefaultExecutionSpace``, or
``-1`` if only host backends are enabled.

Interface
---------

.. cpp:function:: [[nodiscard]] int device_id() noexcept

   :return: The id of the device used by ``DefaultExecutionSpace``, or ``-1``
     if only host backends are enabled.

   .. versionadded:: 4.1


Example
-------

.. code-block:: cpp

   #include <Kokkos_Core.hpp>
   #include <iostream>

   int main(int argc, char* argv[]) {
       Kokkos::initialize(argc, argv);
       {
         std::cout << "device_id: " << Kokkos::device_id() << '\n';
       }
       Kokkos::finalize();
   }


See Also
--------
.. seealso::

   :doc:`num_devices`
      Returns the number of devices available to Kokkos

   :doc:`num_threads`
      Returns the number of threads used by Kokkos

   :doc:`print_configuration`
      Prints Kokkos configuration information to an output stream

   :doc:`../initialize_finalize/initialize`
     Initialize the Kokkos execution environment

   :doc:`../initialize_finalize/InitializationSettings`
     Settings for initializing Kokkos
