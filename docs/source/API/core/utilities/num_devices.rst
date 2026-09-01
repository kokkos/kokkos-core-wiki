``num_devices``
===============

Defined in header ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

    Kokkos::num_devices();

Returns the number of available devices on the system, or ``-1`` if only host
backends are enabled.

Interface
---------

.. cpp:function:: [[nodiscard]] int num_devices() noexcept

   :return: The number of devices available to Kokkos, or ``-1`` if only host
     backends are enabled.

   .. versionadded:: 4.3


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


Notes
-----

.. note::
   :cpp:func:`num_devices` is one of the few runtime functions that may be
   called before :cpp:func:`initialize` or after :cpp:func:`finalize`.


See Also
--------
.. seealso::

   :doc:`device_id`
      Returns the id of the device used by Kokkos

   :doc:`num_threads`
      Returns the number of threads used by Kokkos

   :doc:`print_configuration`
      Prints Kokkos configuration information to an output stream

   :doc:`../initialize_finalize/initialize`
     Initialize the Kokkos execution environment

   :doc:`../initialize_finalize/InitializationSettings`
     Settings for initializing Kokkos
