Version Macros
==============

.. list-table::
   :header-rows: 1
   :align: left

   * - Version integer macros
     - Example value assuming Kokkos version 4.2.1
   * - ``KOKKOS_VERSION``
     - ``40201``
   * - ``KOKKOS_VERSION_MAJOR``
     - ``4``
   * - ``KOKKOS_VERSION_MINOR``
     - ``2``
   * - ``KOKKOS_VERSION_PATCH``
     - ``1``

Kokkos version macros are defined as integers.
``KOKKOS_VERSION`` is equal to ``<MAJOR>*10000 + <MINOR>*100 + <PATCH>``.
``40201`` corresponds to the Kokkos 4.2.1 release version.
A ``99`` patch number denotes a development version.  That is, ``40199``
corresponds to the Kokkos development version post release 4.1

.. code-block:: c++

   #include <Kokkos_Core.hpp>
   #include <iostream>
   static_assert(KOKKOS_VERSION_MAJOR == KOKKOS_VERSION / 10000);
   static_assert(KOKKOS_VERSION_MINOR == KOKKOS_VERSION / 100 % 100);
   static_assert(KOKKOS_VERSION_PATCH == KOKKOS_VERSION % 100);
   #if KOKKOS_VERSION >= 30700
   // Kokkos version is at least 3.7
   #endif
   int main() {
     if (KOKKOS_VERSION_PATCH == 99)
       std::cout << "using development version of Kokkos"
     // ...
   }

.. warning:: Until Kokkos 4.1, ``KOKKOS_VERSION_MINOR`` and ``KOKKOS_VERSION_PATCH`` were not defined when they were meant to be ``0``

Comparison helpers
------------------

.. list-table::
   :header-rows: 1
   :align: left

   * - Function-like version helper macros (since Kokkos 4.1)
   * - ``KOKKOS_VERSION_LESS(major, minor, patch)``
   * - ``KOKKOS_VERSION_LESS_EQUAL(major, minor, patch)``
   * - ``KOKKOS_VERSION_GREATER(major, minor, patch)``
   * - ``KOKKOS_VERSION_GREATER_EQUAL(major, minor, patch)``
   * - ``KOKKOS_VERSION_EQUAL(major, minor, patch)``

``KOKKOS_VERSION_<COMPARE>`` function-like macros return the result of
comparing the version of Kokkos currently used against the version specified by
the three arguments ``major.minor.patch``.  These are available since Kokkos
4.1

.. code-block:: c++

   #include <Kokkos_Core.hpp>
   // sanity check for illustration
   static_assert(KOKKOS_VERSION_EQUAL(KOKKOS_VERSION_MAJOR,
                                      KOKKOS_VERSION_MINOR,
                                      KOKKOS_VERSION_PATCH));
   // set the minimum required version of Kokkos for a project or a file
   static_assert(KOKKOS_VERSION_GREATER_EQUAL(4, 5, 0));

   void do_work() {
     #if KOKKOS_VERSION_GREATER_EQUAL(4, 3, 0)
     // using the new rad functionality
     #else
     // falling back to the old boring stuff
     #endif
   }
