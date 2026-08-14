``printf``
==========

.. role:: cpp(code)
    :language: cpp

Defined in header ``<Kokkos_Printf.hpp>``:sup:`since Kokkos 4.2` which is included from ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

   Kokkos::printf("Hello World!");
   Kokkos::printf("Pi is approx %.2f\n", 3.14159);
   Kokkos::printf("Value at index %d: %f\n", i, data[i]);

Prints formatted output to the standard output stream (``stdout``). This
function can be called from both host and device code, including within
parallel kernels. The behavior is analogous to ``std::printf``, but returns
``void`` instead of an integer to ensure consistent behavior across backends.

Interface
---------

.. cpp:function:: template <typename... Args> KOKKOS_FUNCTION void printf(const char* format, Args... args);

   .. versionadded:: 4.2

   :param format: Null-terminated string specifying how to format the output,
     using format specifiers compatible with C ``printf``
   :param args: Values to be printed according to the format string
   :returns: void (unlike ``std::printf`` which returns the number of
     characters written)

Format Specifiers
^^^^^^^^^^^^^^^^^

Supports standard C ``printf`` format specifiers:

* ``%d``, ``%i`` - signed integer
* ``%u`` - unsigned integer
* ``%f`` - floating point
* ``%e``, ``%E`` - scientific notation
* ``%g``, ``%G`` - shortest representation
* ``%s`` - string
* ``%p`` - pointer
* ``%x``, ``%X`` - hexadecimal
* Width, precision, and length modifiers (e.g., ``%.2f``, ``%10d``)

Example
-------

Basic Usage
^^^^^^^^^^^
.. code-block:: cpp

    #include <Kokkos_Core.hpp>

    int main(int argc, char* argv[]) {
        Kokkos::initialize(argc, argv);
        {
          Kokkos::printf("Starting computation\n");       

          Kokkos::parallel_for("hello", 4, KOKKOS_LAMBDA(int i) {
              Kokkos::printf("hello world from thread %d\n", i);
          });
          Kokkos::fence();

          Kokkos::printf("Computation complete\n");
        }
        Kokkos::finalize();
    }

Debugging Values
^^^^^^^^^^^^^^^^

.. code-block:: cpp

    Kokkos::parallel_for("debug", n, KOKKOS_LAMBDA(int i) {
        if (i < 5) {  // Limit output for large arrays
            Kokkos::printf("data[%d] = %.6f\n", i, data(i));
        }
        if (data(i) < 0) {
            Kokkos::printf("Warning: negative value at index %d: %f\n", i, data(i));
        }
    });

Notes
-----
.. warning::
   
   Calling :cpp:func:`printf` from a kernel may affect register usage and
   reduce performance. Use sparingly in performance-critical code.

See also
--------

.. seealso::

    :doc:`abort`
       Causes abnormal program termination with an error message
    
    :doc:`assert`
       Conditionally aborts if a condition is false; useful for debugging
