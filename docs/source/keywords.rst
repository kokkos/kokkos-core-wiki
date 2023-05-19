.. include:: mydefs.rst

CMake Keywords
##############

.. important::

   With version 3.0 all Kokkos CMake keywords are prefixed with `Kokkos_` which is case-sensitive.

   Recall that to set a keyword in CMake you used the syntax ``-Dkeyword_name=value``.


This page is organized in four sections:

- :ref:`keywords_backends`
- :ref:`keywords_enable_options`
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
    :widths: 25 65 35
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
      * (see below)

    * * ``Kokkos_ENABLE_CUDA_LDG_INTRINSIC``
      * Use CUDA LDG intrinsics
      * ``OFF``

    * * ``Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE``
      * Enable relocatable device code (RDC) for CUDA
      * ``OFF``

    * * ``Kokkos_ENABLE_CUDA_UVM`` :red:`[Deprecated since 4.0]` see `Transition to alternatives <usecases/Moving_from_EnableUVM_to_SharedSpace.html>`_
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

    * * ``Kokkos_ENABLE_HIP_MULTIPLE_KERNEL_INSTANTIATIONS``
      * Instantiate multiple kernels at compile time - improve performance but increase compile time
      * ``OFF``

    * * ``Kokkos_ENABLE_HIP_RELOCATABLE_DEVICE_CODE``
      * Enable relocatable device code (RDC) for HIP
      * ``OFF``

    * * ``Kokkos_ENABLE_LARGE_MEM_TESTS``
      * Perform extra large memory tests
      * ``OFF``

    * * ``Kokkos_ENABLE_PROFILING_LOAD_PRINT``
      * Print information about which profiling tools got loaded
      * ``OFF``

    * * ``Kokkos_ENABLE_TESTS``
      * Build tests
      * ``OFF``

    * * ``Kokkos_ENABLE_TUNING``
      * Create bindings for tuning tools
      * ``OFF``
       

``Kokkos_ENABLE_CUDA_LAMBDA`` default value is ``OFF`` until 3.7 and ``ON`` since 4.0

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

    * * ``Kokkos_ARCH_ADA89``
      * Optimize for the NVIDIA Ada generation CC 8.9 :sup:`since Kokkos 4.1`
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

    * * ``Kokkos_ARCH_HOPPER90``
      * Optimize for the NVIDIA Hopper generation CC 9.0 :sup:`since Kokkos 4.0`
      * ``OFF``

    * * ``Kokkos_ARCH_HSW``
      * Optimize for HSW architecture
      * ``OFF``

    * * ``Kokkos_ARCH_INTEL_GEN``
      * Optimize for Intel GPUs, Just-In-Time compilation*
      * ``OFF``

    * * ``Kokkos_ARCH_INTEL_DG1``
      * Optimize for Intel Iris XeMAX GPU
      * ``OFF``

    * * ``Kokkos_ARCH_INTEL_GEN9``
      * Optimize for Intel GPU Gen9
      * ``OFF``

    * * ``Kokkos_ARCH_INTEL_GEN11``
      * Optimize for Intel GPU Gen11
      * ``OFF``

    * * ``Kokkos_ARCH_INTEL_GEN12LP``
      * Optimize for Intel GPU Gen12LP
      * ``OFF``

    * * ``Kokkos_ARCH_INTEL_XEHP``
      * Optimize for Intel GPU Xe-HP
      * ``OFF``

    * * ``Kokkos_ARCH_INTEL_PVC``
      * Optimize for Intel GPU Ponte Vecchio/GPU Max
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

    * * ``Kokkos_ARCH_NAVI1030`` :red:`[Since 4.0]`
      * Optimize for AMD GPU V620/W6800 GFX1030
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

    * * ``Kokkos_ARCH_SPR``
      * Optimize for Sapphire Rapids architecture
      * ``OFF``

    * * ``Kokkos_ARCH_TURING75``
      * Optimize for TURING75 architecture
      * ``OFF``

    * * ``Kokkos_ARCH_VEGA900`` :red:`[Removed in 4.0]`
      * Optimize for AMD GPU MI25 GFX900
      * ``OFF``

    * * ``Kokkos_ARCH_VEGA906``
      * Optimize for AMD GPU MI50/MI60 GFX906
      * ``OFF``

    * * ``Kokkos_ARCH_VEGA908``
      * Optimize for AMD GPU MI100 GFX908
      * ``OFF``

    * * ``Kokkos_ARCH_VEGA90A``
      * Optimize for AMD GPU MI200 series GFX90A
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

*) ``Kokkos_ARCH_INTEL_GEN`` enables Just-In-Time compilation for Intel GPUs whereas all the other flags for Intel compilers
request Ahead-Of-Time compilation. Just-In-Time compilation means that the compiler is invoked again when the binaries created
are actually executed and only at that point the architecture to compile for is determined. On the other hand, Ahead-Of-Time
compilation describes the standard model where the compiler is only invoked once to create the binary and the architecture to
compile for is determined before the program is run.
