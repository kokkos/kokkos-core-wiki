``abort``
=========

.. role:: cpp(code)
    :language: cpp

Defined in header ``<Kokkos_Abort.hpp>``:sup:`since Kokkos 4.2` which is included from ``<Kokkos_Core.hpp>``

Usage 
-----

.. code-block:: cpp

    Kokkos::abort("helpful error message");

Causes abnormal program termination and prints an error message to the standard
error stream.
This function can be called from both host and device code, including within
parallel kernels.


Interface
---------

.. cpp:function:: KOKKOS_FUNCTION void abort(const char * msg);

   .. versionadded:: 4.2

   :param msg: Null-terminated string containing the error message to print
     before aborting the process.
   :returns: Does not return


Notes
-----

Version History
^^^^^^^^^^^^^^^
* Available in all Kokkos versions
* The fine-grained header ``<Kokkos_Abort.hpp>`` was added in version 4.2

Backend-Specific Behavior
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning::
   **SYCL Backend:** When calling :cpp:func:`abort` from a parallel region with
   the SYCL backend and ``NDEBUG`` is defined, the function does **not** cause
   abnormal termination. Instead, it prints to the standard output stream and
   continues program execution.
   See :ref:`Known Issues <known-issues-sycl-abort>`.

.. warning::
   **NextSilicon Backend:** When calling :cpp:func:`abort` from a parallel
   region with the NextSilicon backend, the function will cause abnormal
   program termination but will not print the message if the region has been
   offloaded to the accelerator.


Example
-------

.. code-block:: cpp

    KOKKOS_FUNCTION void validate_input(int value) {
      if (value < 0) {
        Kokkos::abort("Error: negative value not allowed");
      }
    }

    // Can be used in parallel regions
    Kokkos::parallel_for("check_data", n, KOKKOS_LAMBDA(int i) {
      if (data(i) > threshold) {
        Kokkos::abort("Data value exceeds threshold");
      }
    });

See also
--------

.. seealso::

   :doc:`assert`
      Conditionally aborts if a condition is false; can be disabled in release builds
   
   :doc:`printf`
      Prints formatted output without terminating execution
