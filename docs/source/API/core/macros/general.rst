General Settings
================

The macros on this page describe how Kokkos was configured. Each one is defined
when the corresponding :doc:`CMake option <../../../get-started/configuration-guide>`
is enabled (see :ref:`macros_configuration_relationship`).

Debugging
---------

.. list-table::
   :header-rows: 1
   :align: left
   :widths: 50 50

   * - Macro
     - Description
   * - ``KOKKOS_ENABLE_DEBUG``
     - Extra debug features are activated.
   * - ``KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK``
     - Extra bounds checks are enabled.

Kokkos applies these checks itself, so they are mostly informational.
See :ref:`debugging` in the Configuration Guide for the corresponding options.

Deprecated code
---------------

.. list-table::
   :header-rows: 1
   :align: left
   :widths: 50 50

   * - Macro
     - Description
   * - ``KOKKOS_ENABLE_DEPRECATED_CODE_<N>``
     - Features deprecated in major release ``<N>`` are still available.
   * - ``KOKKOS_ENABLE_DEPRECATION_WARNINGS``
     - Deprecated features generate deprecation warnings.

``KOKKOS_ENABLE_DEPRECATED_CODE_<N>`` is useful for code that must build with several
Kokkos versions:

.. code-block:: c++

   #include <Kokkos_Core.hpp>

   #ifdef KOKKOS_ENABLE_DEPRECATED_CODE_4
   // Kokkos 4.x deprecated features are still available
   #endif

Tuning and performance
----------------------

.. list-table::
   :header-rows: 1
   :align: left
   :widths: 50 50

   * - Macro
     - Description
   * - ``KOKKOS_ENABLE_TUNING``
     - Bindings for tuning tools are available (see `#2422 <https://github.com/kokkos/kokkos/pull/2422>`_).
   * - ``KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION``
     - Certain dependency assumptions are ignored for aggressive vectorization of internal Kokkos loops.
   * - ``KOKKOS_ENABLE_COMPLEX_ALIGN``
     - Complex types are aligned. This macro has no documented CMake option.

C++ Standard
------------

Exactly one macro is defined, matching the C++ standard Kokkos was compiled with.
Applications should compile with the same standard (and the same compiler) as
Kokkos, so most code has no need to test these macros.

.. list-table::
   :header-rows: 1
   :align: left
   :widths: 30 50 30

   * - Macro
     - Description
     - Note
   * - ``KOKKOS_ENABLE_CXX17``
     - Kokkos was compiled with the C++17 language standard.
     - Removed in Kokkos 5.0
   * - ``KOKKOS_ENABLE_CXX20``
     - Kokkos was compiled with the C++20 language standard.
     -
   * - ``KOKKOS_ENABLE_CXX23``
     - Kokkos was compiled with the C++23 language standard.
     -
   * - ``KOKKOS_ENABLE_CXX26``
     - Kokkos was compiled with the C++26 language standard.
     -

Third-Party Libraries
---------------------

These macros give information about what third-party libraries Kokkos was compiled with.
They correspond to the ``Kokkos_ENABLE_<TPL>`` options listed under
:ref:`keywords_tpls` in the Configuration Guide.

.. list-table::
   :header-rows: 1
   :align: left
   :widths: 30 70

   * - Macro
     - Description
   * - ``KOKKOS_ENABLE_HWLOC``
     - `libhwloc <https://www.open-mpi.org/projects/hwloc/>`_ is enabled for NUMA and architecture information.
   * - ``KOKKOS_ENABLE_LIBDL``
     - Kokkos links to the dynamic linker (libdl).
   * - ``KOKKOS_ENABLE_LIBQUADMATH``
     - Kokkos links to the `GCC Quad-Precision Math Library API <https://gcc.gnu.org/onlinedocs/libquadmath/>`_.
   * - ``KOKKOS_ENABLE_ONEDPL``
     - Kokkos links to the `oneDPL library <https://github.com/oneapi-src/oneDPL>`_ when using the SYCL backend.
       Enabling this TPL might affect performance for Kokkos algorithms that use it, e.g., ``sort``.
   * - ``KOKKOS_ENABLE_ROCTHRUST``
     - Kokkos links to the `rocThrust library <https://github.com/ROCm/rocThrust>`_ when using the HIP backend.
       Enabling this TPL might affect performance for Kokkos algorithms that use it, e.g., ``sort``.
