
Keywords and Macros
###################

Device Backends
===============

.. important:: 

    Device backends can be enabled by specifying ``-DKokkos_ENABLE_X``.


.. list-table::
    :widths: 25 60 15
    :header-rows: 1
    :align: left

    * - 
      - Description/info
      - Default

    * - ``Kokkos_ENABLE_CUDA``
      - Whether to build CUDA backend
      - ``OFF``

    * - ``Kokkos_ENABLE_HPX``
      - (Experimental) Whether to build HPX backend 
      - ``OFF``

    * - ``Kokkos_ENABLE_OPENMP``
      - Whether to build OpenMP backend
      - ``OFF``

    * - ``Kokkos_ENABLE_PTHREAD``
      - Whether to build Pthread backend
      - ``OFF``

    * - ``Kokkos_ENABLE_SERIAL``
      - Whether to build serial backend
      - ``ON``

    * - ``Kokkos_ENABLE_HIP``
      - (Experimental) Whether to build HIP backend
      - ``OFF``

    * - ``Kokkos_ENABLE_OPENMPTARGET`` 
      - (Experimental) Whether to build the OpenMP target backend
      - ``OFF``



Enable Options
===============

.. important:: 

    Options can be enabled by specifying ``-DKokkos_ENABLE_X``.


.. list-table::
    :widths: 25 60 15
    :header-rows: 1
    :align: left

    * - 
      - Description/info
      - Default

    * * ``Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION``
      * Whether to aggressively vectorize loops
      * ``OFF``

    * * ``Kokkos_ENABLE_COMPILER_WARNINGS``
      * Whether to print all compiler warnings
      * ``OFF``

    * * ``Kokkos_ENABLE_CUDA_CONSTEXPR``
      * Whether to activate experimental relaxed constexpr functions
      * ``OFF``

    * * ``Kokkos_ENABLE_CUDA_LAMBDA``
      * Whether to activate experimental lambda features
      * ``OFF``

    * * ``Kokkos_ENABLE_CUDA_LDG_INTRINSIC``
      * Whether to use CUDA LDG intrinsics
      * ``OFF``

    * * ``Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE``
      * Whether to enable relocatable device code (RDC) for CUDA
      * ``OFF``

    * * ``Kokkos_ENABLE_CUDA_UVM``
      * Whether to use unified memory (UM) by default for CUDA
      * ``OFF``

    * * ``Kokkos_ENABLE_DEBUG``
      * Whether to activate extra debug features - may increase compile times
      * ``OFF``

    * * ``Kokkos_ENABLE_DEBUG_BOUNDS_CHECK``
      * Whether to use bounds checking - will increase runtime
      * ``OFF``

    * * ``Kokkos_ENABLE_DEBUG_DUALVIEW_MODIFY_CHECK``
      * Debug check on dual views
      * ``OFF``

    * * ``Kokkos_ENABLE_EXAMPLES``
      * Whether to enable building examples
      * ``OFF``

    * * ``Kokkos_ENABLE_HPX_ASYNC_DISPATCH``
      * Whether HPX supports asynchronous dispatch
      * ``OFF``

    * * ``Kokkos_ENABLE_LARGE_MEM_TESTS``
      * Whether to perform extra large memory tests
      * ``OFF``

    * * ``Kokkos_ENABLE_PROFILING_LOAD_PRINT``
      * Whether to print information about which profiling tools gotloaded
      * ``OFF``

    * * ``Kokkos_ENABLE_TESTS``
      * Whether to build serial  backend
      * ``OFF``



Other Options
=============

.. list-table::
    :widths: 25 60 15
    :header-rows: 1
    :align: left

    * - 
      - Description/info
      - Default

    * * ``Kokkos_CXX_STANDARD``
      * The C++ standard for Kokkos to use: c++14, c++17, or c++20. This should be given in CMake style as 14, 17, or 20.
      * STRING Default: 14


Third-party Libraries (TPLs)
============================

The following options control enabling TPLs:

.. list-table::
    :widths: 25 60 15
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
    :widths: 25 60 15
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
    :widths: 25 60 15
    :header-rows: 1
    :align: left

    * - 
      - Description/info
      - Default

    * * ``HPX_DIR`` or ``HPX_ROOT``
      * Location of HPX prefix (ROOT) or CMake config file (DIR)
      * PATH Default:


Architecture Keywords
=====================

.. important:: 

    Architecture-specific optimizations can be enabled by specifying ``-DKokkos_ARCH_X``.


.. list-table::
    :widths: 25 60 15
    :header-rows: 1
    :align: left

    * - 
      - Description/info
      - Default

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

    * * ``Kokkos_ARCH_ARMV8_TX2``
      * Optimize for ARMV8_TX2 architecture
      * ``OFF``

    * * ``Kokkos_ARCH_BDW``
      * Optimize for BDW architecture
      * ``OFF``

    * * ``Kokkos_ARCH_BGQ``
      * Optimize for BGQ architecture
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

    * * ``Kokkos_ARCH_HSW``
      * Optimize for HSW architecture
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

    * * ``Kokkos_ARCH_VOLTA70``
      * Optimize for VOLTA70 architecture
      * ``OFF``

    * * ``Kokkos_ARCH_VOLTA72``
      * Optimize for VOLTA72 architecture
      * ``OFF``

    * * ``Kokkos_ARCH_WSM``
      * Optimize for WSM architecture
      * ``OFF``
