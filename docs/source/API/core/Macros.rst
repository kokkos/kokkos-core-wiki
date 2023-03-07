Macros
======

Version Macros
--------------

+--------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Macro              | Description                                                                                                                                                                  |
+====================+==============================================================================================================================================================================+
| ``KOKKOS_VERSION`` | The Kokkos version; ``KOKKOS_VERSION % 100`` is the patch level, ``KOKKOS_VERSION / 100 % 100`` is the minor version, and ``KOKKOS_VERSION / 10000`` is the major version.   |
+--------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

General Settings
----------------

+-------------------------------------------------+-------------------------------------------------------------------------------------------------------------+
| Macro                                           | Description                                                                                                 |
+=================================================+=============================================================================================================+
| ``KOKKOS_ENABLE_DEBUG``                         | Defined if extra debug features are activated.                                                              |
+-------------------------------------------------+-------------------------------------------------------------------------------------------------------------+
| ``KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK``            | Defined if extra bounds checks are enabled.                                                                 |
+-------------------------------------------------+-------------------------------------------------------------------------------------------------------------+
| ``KOKKOS_ENABLE_DEBUG_DUALVIEW_MODIFY_CHECK``   | Defined if debug checks for ``Kokkos::DualView`` objects are enabled.                                       |
+-------------------------------------------------+-------------------------------------------------------------------------------------------------------------+
| ``KOKKOS_ENABLE_DEPRECATED_CODE_3``             | Defined if features deprecated in major release 3 are still available.                                      |
+-------------------------------------------------+-------------------------------------------------------------------------------------------------------------+
| ``KOKKOS_ENABLE_DEPRECATION_WARNING``           | Defined if deprecated features generate deprecation warnings.                                               |
+-------------------------------------------------+-------------------------------------------------------------------------------------------------------------+
| ``KOKKOS_ENABLE_HBWSPACE``                      | Defined if the experimental ``HBWSpace`` memory space is enabled, enabled by KOKKOS_ENABLE_MEMKIND.         |
+-------------------------------------------------+-------------------------------------------------------------------------------------------------------------+
| ``KOKKOS_ENABLE_PROFILING_LOAD_PRINT``          | Kokkos will output a message when the profiling library is loaded.                                          |
+-------------------------------------------------+-------------------------------------------------------------------------------------------------------------+
| ``KOKKOS_ENABLE_TUNING``                        | Whether bindings for tunings are available (see `#2422 <https://github.com/kokkos/kokkos/pull/2422>`_).     |
+-------------------------------------------------+-------------------------------------------------------------------------------------------------------------+
| ``KOKKOS_ENABLE_COMPLEX_ALIGN``                 | Whether complex types are aligned.                                                                          |
+-------------------------------------------------+-------------------------------------------------------------------------------------------------------------+
| ``KOKKOS_ENABLE_AGGRESSIVE_VECTORIZATION``      | Whether certain dependency assumptions are ignored for aggressive vectorization of internal Kokkos loops.   |
+-------------------------------------------------+-------------------------------------------------------------------------------------------------------------+

Execution Spaces
----------------

The following macros can be used to test whether or not a specified execution space
is enabled. They can be tested for existence (e.g. ``#ifdef KOKKOS_ENABLE_SERIAL``).

.. _Serial: execution_spaces.html

.. |Serial| replace:: :cppkokkos:func:`Serial`

.. _OpenMP: execution_spaces.html

.. |OpenMP| replace:: :cppkokkos:func:`OpenMP`

.. _Threads: execution_spaces.html

.. |Threads| replace:: :cppkokkos:func:`Threads`

.. _Cuda: execution_spaces.html

.. |Cuda| replace:: :cppkokkos:func:`Cuda`

.. _HPX: execution_spaces.html

.. |HPX| replace:: :cppkokkos:func:`HPX`

+--------------------------------+--------------------------------------------------------------------------+
| Macro                          | Description                                                              |
+================================+==========================================================================+
| ``KOKKOS_ENABLE_SERIAL``       | Defined if the |Serial|_ execution space is enabled.                     |
+--------------------------------+--------------------------------------------------------------------------+
| ``KOKKOS_ENABLE_OPENMP``       | Defined if the |OpenMP|_ execution space is enabled.                     |
+--------------------------------+--------------------------------------------------------------------------+
| ``KOKKOS_ENABLE_OPENMPTARGET`` | Defined if the experimental ``OpenMPTarget`` execution space is enabled. |
+--------------------------------+--------------------------------------------------------------------------+
| ``KOKKOS_ENABLE_THREADS``      | Defined if the |Threads|_ execution space is enabled.                    |
+--------------------------------+--------------------------------------------------------------------------+
| ``KOKKOS_ENABLE_CUDA``         | Defined if the |Cuda|_ execution space is enabled.                       |
+--------------------------------+--------------------------------------------------------------------------+
| ``KOKKOS_ENABLE_HIP``          | Defined if the experimental ``HIP`` execution space is enabled.          |
+--------------------------------+--------------------------------------------------------------------------+
| ``KOKKOS_ENABLE_HPX``          | Defined if the |HPX|_ execution space is enabled.                        |
+--------------------------------+--------------------------------------------------------------------------+
| ``KOKKOS_ENABLE_SYCL``         | Defined if the experimental ``SYCL`` execution space is enabled.         |
+--------------------------------+--------------------------------------------------------------------------+

Backend options
---------------

+-------------------------------------------------------+---------------------------------------------------------------------------------------+
| Macro                                                 | Description                                                                           |
+=======================================================+=======================================================================================+
| ``KOKKOS_ENABLE_CUDA_CONSTEXPR``                      | Defined if the CUDA backend supports constexpr functions.                             |
+-------------------------------------------------------+---------------------------------------------------------------------------------------+
| ``KOKKOS_ENABLE_CUDA_LAMBDA``                         | Defined if the CUDA backend supports lambdas.                                         |
+-------------------------------------------------------+---------------------------------------------------------------------------------------+
| ``KOKKOS_ENABLE_CUDA_LDG_INTRINSINCS``                | Defined if the CUDA backend supports LDG intrinsic.                                   |
+-------------------------------------------------------+---------------------------------------------------------------------------------------+
| ``KOKKOS_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE``        | Defined if the CUDA backend supports relocatable device code.                         |
+-------------------------------------------------------+---------------------------------------------------------------------------------------+
| ``KOKKOS_ENABLE_CUDA_UVM``                            | If defined, the default CUDA memory space is CudaUVMSpace, otherwise it is CudaSpace. |
+-------------------------------------------------------+---------------------------------------------------------------------------------------+
| ``KOKKOS_ENABLE_HIP_MULTIPLE_KERNEL_INSTANTIATIONS``  | If defined, multiple kernel versions are instantiated potentially improving run time. |
+-------------------------------------------------------+---------------------------------------------------------------------------------------+
| ``KOKKOS_ENABLE_HIP_RELOCATABLE_DEVICE_CODE``         | Defined if the HIP backend supports relocatable device code.                          |
+-------------------------------------------------------+---------------------------------------------------------------------------------------+

C++ Standard Settings
---------------------

Kokkos supports the latest C++ language standards. Certain features may use different
code paths or have different compiler support depending on the language standard that
is enabled. The following macros are exposed to determine what language standard
Kokkos was compiled with.

+-------------------------+-----------------------------------------+
| Macro                   | Description                             |
+=========================+=========================================+
| ``KOKKOS_ENABLE_CXX14`` | The C++14 language standard is enabled. |
+-------------------------+-----------------------------------------+
| ``KOKKOS_ENABLE_CXX17`` | The C++17 language standard is enabled. |
+-------------------------+-----------------------------------------+
| ``KOKKOS_ENABLE_CXX20`` | The C++20 language standard is enabled. |
+-------------------------+-----------------------------------------+

Third-Party Library Settings
----------------------------

These defines give information about what third-party libaries Kokkos was compiled with.

+-------------------------------+-----------------------------------------------------------------------------------------------------------------------+
| Macro                         | Description                                                                                                           |
+===============================+=======================================================================================================================+
| ``KOKKOS_ENABLE_HWLOC``       | Defined if `libhwloc <https://www.open-mpi.org/projects/hwloc/>`_ is enabled for NUMA and architecture information.   |
+-------------------------------+-----------------------------------------------------------------------------------------------------------------------+
| ``KOKKOS_ENABLE_LIBRT``       | Defined if Kokkos links to the POSIX librt for backwards compatibility.                                               |
+-------------------------------+-----------------------------------------------------------------------------------------------------------------------+
| ``KOKKOS_ENABLE_MEMKIND``     | Defined if Kokkos enables the `Memkind <https://github.com/memkind/memkind>`_ heap manager, enables HBWSpace.         |
+-------------------------------+-----------------------------------------------------------------------------------------------------------------------+
| ``KOKKOS_ENABLE_LIBDL``       | Defined if Kokkos links to the dynamic linker (libdl).                                                                |
+-------------------------------+-----------------------------------------------------------------------------------------------------------------------+
| ``KOKKOS_ENABLE_LIBQUADMATH`` | Defined if Kokkos links to the `GCC Quad-Precision Math Library API <https://gcc.gnu.org/onlinedocs/libquadmath/>`_.  |
+-------------------------------+-----------------------------------------------------------------------------------------------------------------------+

Architectures
-------------

+-----------------------------------+-----------------------------------------------------------------------------------+
| Macro                             | Description                                                                       |
+===================================+===================================================================================+
| ``KOKKOS_ARCH_SSE42``             | Optimize for SSE 4.2                                                              |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_ARMV80``            | Optimize for ARMv8.0 Compatible CPU (HOST)                                        |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_ARMV8_THUNDERX``    | Optimize for ARMv8 Cavium ThunderX CPU (HOST)                                     |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_ARMV81``            | Optimize for ARMv8.1 Compatible CPU (HOST)                                        |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_ARMV8_THUNDERX2``   | Optimize for ARMv8 Cavium ThunderX2 CPU (HOST)                                    |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_AMD_AVX2``          | Optimize for AVX2 (enabled for Zen)                                               |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_AVX``               | Optimize for AVX                                                                  |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_AVX2``              | Optimize for AVX2                                                                 |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_AVX512XEON``        | Optimize for Skylake(AVX512)                                                      |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_KNC``               | Optimize for Intel Knights Corner Xeon Phi (HOST)                                 |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_AVX512MIC``         | Optimize for Many Integrated Core (MIC; AVX512)                                   |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_POWER7``            | Optimize for IBM POWER7 CPUs (HOST)                                               |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_POWER8``            | Optimize for IBM POWER8 CPUs (HOST)                                               |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_POWER9``            | Optimize for IBM POWER9 CPUs (HOST)                                               |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_INTEL_GEN``         | Optimize for Intel GPUs, Just-In-Time compilation (GPU)                          |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_INTEL_DG1``         | Optimize for Intel Iris XeMAX GPU (GPU)                                           |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_INTEL_GEN9``        | Optimize for Intel GPU Gen9 (GPU)                                                 |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_INTEL_GEN11``       | Optimize for Intel GPU Gen11 (GPU)                                                |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_INTEL_GEN12LP``     | Optimize for Intel GPU Gen12LP (GPU)                                              |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_INTEL_XEHP``        | Optimize for Intel GPU Xe-HP (GPU)                                                |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_INTEL_PVC``         | Optimize for Intel GPU Ponte Vecchio/GPU Max (GPU)                                |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_INTEL_GPU``         | Set if any Intel GPU architecture has been enabled                                |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_KEPLER``            | Set if any NVIDIA Kepler architecture has been enabled                            |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_KEPLER30``          | Optimize for NVIDIA Kepler generation CC 3.0 (GPU)                                |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_KEPLER32``          | Optimize for NVIDIA Kepler generation CC 3.2 (GPU)                                |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_KEPLER35``          | Optimize for NVIDIA Kepler generation CC 3.5 (GPU)                                |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_KEPLER37``          | Optimize for NVIDIA Kepler generation CC 3.7 (GPU)                                |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_MAXWELL``           | Set if any NVIDIA Maxwell architecture has been enabled                           |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_MAXWELL50``         | Optimize for NVIDIA Maxwell generation CC 5.0 (GPU)                               |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_MAXWELL52``         | Optimize for NVIDIA Maxwell generation CC 5.2 (GPU)                               |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_MAXWELL53``         | Optimize for NVIDIA Maxwell generation CC 5.3 (GPU)                               |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_NAVI``              | Set if any AMD Navi GPU architecture as been enabled :sup:`Since Kokkos 4.0`      |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_NAVI1030``          | Optimize for AMD GPU V620/W6800 GFX1030 (GPU) :sup:`Since Kokkos 4.0`             |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_PASCAL``            | Set if any NVIDIA Pascal architecture has been enabled                            |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_PASCAL60``          | Optimize for NVIDIA Pascal generation CC 6.0 (GPU)                                |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_PASCAL61``          | Optimize for NVIDIA Pascal generation CC 6.1 (GPU)                                |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_VOLTA``             | Set if any NVIDIA Volta architecture has been enabled                             |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_VOLTA70``           | Optimize for NVIDIA Volta generation CC 7.0 (GPU)                                 |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_VOLTA72``           | Optimize for NVIDIA Volta generation CC 7.2 (GPU)                                 |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_TURING75``          | Optimize for NVIDIA Turing generation CC 7.5 (GPU)                                |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_AMPERE``            | Set if any NVIDIA Ampere architecture has been enabled                            |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_AMPERE80``          | Optimize for NVIDIA Ampere generation CC 8.0 (GPU)                                |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_AMPERE86``          | Optimize for NVIDIA Ampere generation CC 8.6 (GPU)                                |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_HOPPER``            | Set if any NVIDIA Hopper architecture has been enabled :sup:`since Kokkos 4.0`    |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_HOPPER90``          | Optimize for NVIDIA Hopper generation CC 9.0 (GPU) :sup:`since Kokkos 4.0`        |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_AMD_ZEN``           | Optimize for AMD Zen architecture (HOST)                                          |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_AMD_ZEN2``          | Optimize for AMD Zen2 architecture (HOST)                                         |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_AMD_ZEN3``          | Optimize for AMD Zen3 architecture (HOST)                                         |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_VEGA``              | Set if any AMD Vega GPU architecture as been enabled                              |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_VEGA900``           | Optimize for AMD GPU MI25 GFX900 (GPU) :sup:`Removed in Kokkos 4.0`               |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_VEGA906``           | Optimize for AMD GPU MI50/MI60 GFX906 (GPU)                                       |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_VEGA908``           | Optimize for AMD GPU MI100 GFX908 (GPU)                                           |
+-----------------------------------+-----------------------------------------------------------------------------------+
| ``KOKKOS_ARCH_VEGA90A``           | Optimize for AMD GPU MI200 series GFX90A (GPU)                                    |
+-----------------------------------+-----------------------------------------------------------------------------------+
