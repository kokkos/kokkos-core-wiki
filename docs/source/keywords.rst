.. include:: mydefs.rst

CMake Keywords
##############

.. important::

   With version 3.0 all Kokkos CMake keywords are prefixed with `Kokkos_` which is case-sensitive.

   Recall that to set a keyword in CMake you used the syntax ``-Dkeyword_name``.


This page is organized in four sections:

- :ref:`keywords_backends`
- :ref:`keywords_enable_options`
- :ref:`keywords_enable_other_options`
- :ref:`keywords_tpls`
- :ref:`keywords_arch`

.. _keywords_backends:

Device Backends
===============

.. list-table::
    :widths: 25 65 10
    :header-rows: 1
    :align: left

    * -
      - Description/info
      - Default

    * - ``Kokkos_ENABLE_CUDA``
      - To build CUDA backend
      - ``OFF``

    * - ``Kokkos_ENABLE_OPENMP``
      - To build OpenMP backend
      - ``OFF``

    * - ``Kokkos_ENABLE_THREADS``
      - To build C++ Threads backend
      - ``OFF``

    * - ``Kokkos_ENABLE_SERIAL``
      - To build serial backend
      - ``ON``

    * - ``Kokkos_ENABLE_HIP``
      - To build HIP backend
      - ``OFF``

    * - ``Kokkos_ENABLE_OPENMPTARGET``
      - :red:`[Experimental]` To build the OpenMP target backend
      - ``OFF``

    * - ``Kokkos_ENABLE_SYCL``
      - :red:`[Experimental]` To build SYCL backend
      - ``OFF``

    * - ``Kokkos_ENABLE_HPX``
      - :red:`[Experimental]` To build HPX backend
      - ``OFF``


.. _keywords_enable_options:

Enable Options
===============

.. list-table::
    :widths: 25 65 10
    :header-rows: 1
    :align: left

    * -
      - Description/info
      - Default

    * * ``Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION``
      * Aggressively vectorize loops
      * ``OFF``

    * * ``Kokkos_ENABLE_COMPILER_WARNINGS``
      * Print all compiler warnings
      * ``OFF``

    * * ``Kokkos_ENABLE_CUDA_CONSTEXPR``
      * Activate experimental relaxed constexpr functions
      * ``OFF``

    * * ``Kokkos_ENABLE_CUDA_LAMBDA``
      * Activate experimental lambda features
      * ``OFF``

    * * ``Kokkos_ENABLE_CUDA_LDG_INTRINSIC``
      * Use CUDA LDG intrinsics
      * ``OFF``

    * * ``Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE``
      * Enable relocatable device code (RDC) for CUDA
      * ``OFF``

    * * ``Kokkos_ENABLE_CUDA_UVM``
      * Use unified memory (UM) by default for CUDA
      * ``OFF``

    * * ``Kokkos_ENABLE_DEBUG``
      * Activate extra debug features - may increase compile times
      * ``OFF``

    * * ``Kokkos_ENABLE_DEBUG_BOUNDS_CHECK``
      * Use bounds checking - will increase runtime
      * ``OFF``

    * * ``Kokkos_ENABLE_DEBUG_DUALVIEW_MODIFY_CHECK``
      * Debug check on dual views
      * ``OFF``

    * * ``Kokkos_ENABLE_DEPRECATED_CODE``
      * Enable deprecated code
      * ``OFF``

    * * ``Kokkos_ENABLE_EXAMPLES``
      * Enable building examples
      * ``OFF``

    * * ``Kokkos_ENABLE_MULTIPLE_KERNEL_INSTANTIATIONS``
      * Instantiate multiple kernels at compile time - improve performance but increase compile time
      * ``OFF``

    * * ``Kokkos_ENABLE_HIP_RELOCATABLE_DEVICE_CODE``
      * Enable relocatable device code (RDC) for HIP
      * ``OFF``

    * * ``Kokkos_ENABLE_HPX_ASYNC_DISPATCH``
      * Whether HPX supports asynchronous dispatch
      * ``OFF``

    * * ``Kokkos_ENABLE_LARGE_MEM_TESTS``
      * Perform extra large memory tests
      * ``OFF``

    * * ``Kokkos_ENABLE_PROFILING``
      * Create bindings for profiling tools
      * ``ON``

    * * ``Kokkos_ENABLE_PROFILING_LOAD_PRINT``
      * Print information about which profiling tools got loaded
      * ``OFF``

    * * ``Kokkos_ENABLE_TESTS``
      * Build tests
      * ``OFF``

.. _keywords_enable_other_options:

Other Options
=============

.. list-table::
    :widths: 25 50 25
    :header-rows: 1
    :align: left

    * -
      - Description/info
      - Default

    * * ``Kokkos_CXX_STANDARD``
      * The C++ standard for Kokkos to use: c++14, c++17, or c++20. This should be given in CMake style as 14, 17, or 20.
      * STRING Default: 14

.. _keywords_tpls:

Third-party Libraries (TPLs)
============================

The following options control enabling TPLs:

.. list-table::
    :widths: 30 60 10
    :header-rows: 1
    :align: left

    * -
      - Description/info
      - Default

    * * ``Kokkos_ENABLE_HPX``
      * Whether to enable the HPX library
      * ``OFF``
    * * ``Kokkos_ENABLE_HWLOC``
      * Whether to enable the HWLOC library
      * ``Off``
    * * ``Kokkos_ENABLE_LIBNUMA``
      * Whether to enable the LIBNUMA library
      * ``Off``
    * * ``Kokkos_ENABLE_MEMKIND``
      * Whether to enable the MEMKIND library
      * ``Off``
    * * ``Kokkos_ENABLE_LIBDL``
      * Whether to enable the LIBDL library
      * ``On``
    * * ``Kokkos_ENABLE_LIBRT``
      * Whether to enable the LIBRT library
      * ``Off``

The following options control finding and configuring non-CMake TPLs:

.. list-table::
    :widths: 35 45 20
    :header-rows: 1
    :align: left

    * -
      - Description/info
      - Default

    * * ``Kokkos_CUDA_DIR`` or ``CUDA_ROOT``
      * Location of CUDA install prefix for libraries
      * PATH Default:

    * * ``Kokkos_HWLOC_DIR`` or ``HWLOC_ROOT``
      * Location of HWLOC install prefix
      * PATH Default:

    * * ``Kokkos_LIBNUMA_DIR`` or ``LIBNUMA_ROOT``
      * Location of LIBNUMA install prefix
      * PATH Default:

    * * ``Kokkos_MEMKIND_DIR`` or ``MEMKIND_ROOT``
      * Location of MEMKIND install prefix
      * PATH Default:

    * * ``Kokkos_LIBDL_DIR`` or ``LIBDL_ROOT``
      * Location of LIBDL install prefix
      * PATH Default:

    * * ``Kokkos_LIBRT_DIR`` or ``LIBRT_ROOT``
      * Location of LIBRT install prefix
      * PATH Default:

The following options control ``find_package`` paths for CMake-based TPLs:

.. list-table::
    :widths: 35 60 25
    :header-rows: 1
    :align: left

    * -
      - Description/info
      - Default

    * * ``HPX_DIR`` or ``HPX_ROOT``
      * Location of HPX prefix (ROOT) or CMake config file (DIR)
      * PATH Default:

.. _keywords_arch:

Architecture Keywords
=====================

.. list-table::
    :widths: 25 65 10
    :header-rows: 1
    :align: left

    * -
      - Description/info
      - Default

    * * ``Kokkos_ARCH_NATIVE``
      * Optimize for the local CPU architecture
      * ``OFF``

    * * ``Kokkos_ARCH_A64FX``
      * Optimize for ARMv8.2 with SVE Support
      * ``OFF``

    * * ``Kokkos_ARCH_AMPERE80``
      * Optimize for the NVIDIA Ampere generation CC 8.0
      * ``OFF``

    * * ``Kokkos_ARCH_AMPERE86``
      * Optimize for the NVIDIA Ampere generation CC 8.6
      * ``OFF``

    * * ``Kokkos_ARCH_AMDAVX``
      * Optimize for AMDAVX architecture
      * ``OFF``

    * * ``Kokkos_ARCH_ARMV80``
      * Optimize for ARMV80 architecture
      * ``OFF``

    * * ``Kokkos_ARCH_ARMV81``
      * Optimize for ARMV81 architecture
      * ``OFF``

    * * ``Kokkos_ARCH_ARMV8_THUNDERX``
      * Optimize for ARMV8_THUNDERX architecture
      * ``OFF``

    * * ``Kokkos_ARCH_ARMV8_THUNDERX2``
      * Optimize for the ARMV8_TX2 architecture
      * ``OFF``

    * * ``Kokkos_ARCH_ARMV8_TX2``
      * Optimize for ARMV8_TX2 architecture
      * ``OFF``

    * * ``Kokkos_ARCH_BDW``
      * Optimize for BDW architecture
      * ``OFF``

    * * ``Kokkos_ARCH_BGQ``
      * Optimize for BGQ architecture
      * ``OFF``

    * * ``Kokkos_ARCH_HSW``
      * Optimize for HSW architecture
      * ``OFF``

    * * ``Kokkos_ARCH_INTEL_GEN``
      * Optimize for Intel GPUs Gen9+
      * ``OFF``

    * * ``Kokkos_ARCH_KEPLER30``
      * Optimize for KEPLER30 architecture
      * ``OFF``

    * * ``Kokkos_ARCH_KEPLER32``
      * Optimize for KEPLER32 architecture
      * ``OFF``

    * * ``Kokkos_ARCH_KEPLER35``
      * Optimize for KEPLER35 architecture
      * ``OFF``

    * * ``Kokkos_ARCH_KEPLER37``
      * Optimize for KEPLER37 architecture
      * ``OFF``

    * * ``Kokkos_ARCH_KNC``
      * Optimize for KNC architecture
      * ``OFF``

    * * ``Kokkos_ARCH_KNL``
      * Optimize for KNL architecture
      * ``OFF``

    * * ``Kokkos_ARCH_MAXWELL50``
      * Optimize for MAXWELL50 architecture
      * ``OFF``

    * * ``Kokkos_ARCH_MAXWELL52``
      * Optimize for MAXWELL52 architecture
      * ``OFF``

    * * ``Kokkos_ARCH_MAXWELL53``
      * Optimize for MAXWELL53 architecture
      * ``OFF``

    * * ``Kokkos_ARCH_PASCAL60``
      * Optimize for PASCAL60 architecture
      * ``OFF``

    * * ``Kokkos_ARCH_PASCAL61``
      * Optimize for PASCAL61 architecture
      * ``OFF``

    * * ``Kokkos_ARCH_POWER7``
      * Optimize for POWER7 architecture
      * ``OFF``

    * * ``Kokkos_ARCH_POWER8``
      * Optimize for POWER8 architecture
      * ``OFF``

    * * ``Kokkos_ARCH_POWER9``
      * Optimize for POWER9 architecture
      * ``OFF``

    * * ``Kokkos_ARCH_SKX``
      * Optimize for SKX architecture
      * ``OFF``

    * * ``Kokkos_ARCH_SNB``
      * Optimize for SNB architecture
      * ``OFF``

    * * ``Kokkos_ARCH_TURING75``
      * Optimize for TURING75 architecture
      * ``OFF``

    * * ``Kokkos_ARCH_VEGA900``
      * Optimize for AMD GPU MI25 GFX900
      * ``OFF``

    * * ``Kokkos_ARCH_VEGA906``
      * Optimize for AMD GPU MI50/MI60 GFX906
      * ``OFF``

    * * ``Kokkos_ARCH_VEGA908``
      * Optimize for AMD GPU MI100 GFX908
      * ``OFF``

    * * ``Kokkos_ARCH_VOLTA70``
      * Optimize for VOLTA70 architecture
      * ``OFF``

    * * ``Kokkos_ARCH_VOLTA72``
      * Optimize for VOLTA72 architecture
      * ``OFF``

    * * ``Kokkos_ARCH_WSM``
      * Optimize for WSM architecture
      * ``OFF``

    * * ``Kokkos_ARCH_ZEN``
      * Optimize for Zen architecture
      * ``OFF``

    * * ``Kokkos_ARCH_ZEN2``
      * Optimize for Zen2 architecture
      * ``OFF``

    * * ``Kokkos_ARCH_ZEN3``
      * Optimize for Zen3 architecture
      * ``OFF``
