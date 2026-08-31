``print_configuration``
=======================

Defined in header ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

    Kokkos::print_configuration(std::cout);
    Kokkos::print_configuration(output_stream, /*verbose=*/ true);

Prints Kokkos configuration information to an output stream.
This includes both compile-time configuration details (enabled backends,
compiler settings, version information, build configuration) and runtime
information determined during :cpp:func:`initialize` (such as number of
threads for host parallel backends, or number of visible devices and device IDs
for device backends).

API Reference
-------------

.. cpp:function:: void print_configuration(std::ostream& os, bool verbose = false)

   Prints Kokkos configuration information to the specified output stream.

   :param os: Output stream to write configuration information to
   :param verbose: If ``true``, prints additional detailed information


Example
-------

.. code-block:: cpp

   #include <Kokkos_Core.hpp>
   #include <iostream>
   
   int main(int argc, char* argv[]) {
       Kokkos::initialize(argc, argv);
       {
       
         // Print basic configuration to the standard output
         Kokkos::print_configuration(std::cout);

         // Write vebose configuration to a log file
         std::ofstream log_file("kokkos_config.log");
         if (log_file.is_open()) {
             Kokkos::print_configuration(log_file, /*verbose=*/ true);
             log_file.close();
         }
       
       }
       Kokkos::finalize();
   }

Notes
-----

.. warning::
   Kokkos makes no guarantees about the format of the output. The format is subject to change between releases.

.. tip::
   You can print the configuration to standard output without adding an
   explicit :cpp:func:`print_configuration` call and recompiling by setting the
   environment variable ``KOKKOS_PRINT_CONFIGURATION=1`` before running your
   application.

See Also
--------
.. seealso::

   :doc:`device_id`
      Returns the id of the device used by Kokkos

   :doc:`num_devices`
      Returns the number of devices available to Kokkos

   :doc:`num_threads`
      Returns the number of threads used by Kokkos

   :doc:`../initialize_finalize/initialize`
     Initialize the Kokkos execution environment
