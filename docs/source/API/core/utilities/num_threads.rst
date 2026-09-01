``num_threads``
===============

Defined in header ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

    Kokkos::num_threads();

Returns the number of concurrent threads that are used by
``DefaultHostExecutionSpace``.

Interface
---------

.. cpp:function:: [[nodiscard]] int num_threads() noexcept

   :return: The number of concurrent threads used by
     ``DefaultHostExecutionSpace``.

   .. versionadded:: 4.1


Example
-------

.. code-block:: cpp

   #include <Kokkos_Core.hpp>
   #include <iostream>

   int main(int argc, char* argv[]) {
       Kokkos::initialize(argc, argv);
       {
         std::cout << "num_threads: " << Kokkos::num_threads() << '\n';
       }
       Kokkos::finalize();
   }


See Also
--------
.. seealso::

   :doc:`device_id`
      Returns the id of the device used by Kokkos

   :doc:`num_devices`
      Returns the number of devices available to Kokkos

   :doc:`print_configuration`
      Prints Kokkos configuration information to an output stream

   :doc:`../initialize_finalize/initialize`
     Initialize the Kokkos execution environment

   :doc:`../initialize_finalize/InitializationSettings`
     Settings for initializing Kokkos
