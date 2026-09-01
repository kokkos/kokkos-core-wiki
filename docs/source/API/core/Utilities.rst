Utilities
=========


Error Handling and Diagnostics
------------------------------
Kokkos provides utility functions for error handling, debugging, and output
that work consistently across host and device code.

.. list-table::
   :align: left
   :widths: 30 70

   * - :doc:`utilities/abort`
     - Terminates the program immediately with an error message.
   * - :doc:`utilities/assert`
     - Aborts the program if the user-specified condition is not ``true``.
       Behavior depends on ``NDEBUG`` and ``KOKKOS_ENABLE_DEBUG`` macros.
   * - :doc:`utilities/printf`
     - Prints formatted output to the standard output stream.

.. toctree::
   :hidden:
   :maxdepth: 1

   ./utilities/abort
   ./utilities/assert
   ./utilities/printf


Runtime and Device Information
------------------------------
Kokkos provides utility functions to query runtime information about the
Kokkos execution environment.

.. list-table::
   :align: left
   :widths: 30 70

   * - :doc:`utilities/device_id`
     - Returns the id of the device used by ``DefaultExecutionSpace``.
   * - :doc:`utilities/num_devices`
     - Returns the number of devices available to Kokkos.
   * - :doc:`utilities/num_threads`
     - Returns the number of threads used by ``DefaultHostExecutionSpace``.
   * - :doc:`utilities/print_configuration`
     - Prints the Kokkos compile-time and runtime configuration to an output stream.

.. toctree::
   :hidden:
   :maxdepth: 1

   ./utilities/device_id
   ./utilities/num_devices
   ./utilities/num_threads
   ./utilities/print_configuration


Other
-----
.. toctree::
   :maxdepth: 1

   ./utilities/min_max_clamp
   ./utilities/swap
   ./utilities/timer
