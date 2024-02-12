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

Kokkos backends
===============

Serial backend
--------------

.. list-table::
    :widths: 25 65 10
    :header-rows: 1
    :align: left

    * -
      - Description/info
      - Default

    * - ``Kokkos_ENABLE_SERIAL``
      - To build Serial backend targeting CPUs
      - ``OFF``

Host parallel backends
----------------------

.. list-table::
    :widths: 25 65 10
    :header-rows: 1
    :align: left

    * -
      - Description/info
      - Default

    * - ``Kokkos_ENABLE_OPENMP``
      - To build OpenMP backend
      - ``OFF``

    * - ``Kokkos_ENABLE_THREADS``
      - To build C++ Threads backend
      - ``OFF``

    * - ``Kokkos_ENABLE_HPX``
      - :red:`[Experimental]` To build HPX backend
      - ``OFF``

Device backends
---------------

.. list-table::
    :widths: 25 65 10
    :header-rows: 1
    :align: left

    * -
      - Description/info
      - Default

    * - ``Kokkos_ENABLE_CUDA``
      - To build CUDA backend targeting NVIDIA GPUs
      - ``OFF``

    * - ``Kokkos_ENABLE_HIP``
      - To build HIP backend targeting AMD GPUs
      - ``OFF``

    * - ``Kokkos_ENABLE_SYCL``
      - :red:`[Experimental]` To build SYCL backend
      - ``OFF``

    * - ``Kokkos_ENABLE_OPENMPTARGET``
      - :red:`[Experimental]` To build the OpenMP target backend
      - ``OFF``


.. _keywords_enable_options:

Options
=======

General options
---------------

.. list-table::
    :widths: 25 65 35
    :header-rows: 1
    :align: left

    * -
      - Description/info
      - Default

    * * ``Kokkos_ENABLE_BENCHMARKS``
      * Build benchmarks
      * ``OFF``

    * * ``Kokkos_ENABLE_EXAMPLES``
      * Build examples
      * ``OFF``

    * * ``Kokkos_ENABLE_TESTS``
      * Build tests
      * ``OFF``

    * * ``Kokkos_ENABLE_TUNING``
      * Create bindings for tuning tools
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

    * * ``Kokkos_ENABLE_DEPRECATED_CODE_3``
      * Enable deprecated code in the Kokkos 3.x series
      * ``OFF``

    * * ``Kokkos_ENABLE_DEPRECATED_CODE_4``
      * Enable deprecated code in the Kokkos 4.x series
      * ``ON``

    * * ``Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION``
      * Aggressively vectorize loops
      * ``OFF``

    * * ``Kokkos_ENABLE_COMPILER_WARNINGS``
      * Print all compiler warnings
      * ``OFF``

    * * ``Kokkos_ENABLE_HEADER_SELF_CONTAINMENT_TESTS``
      * Check that headers are self-contained
      * ``OFF``

    * * ``Kokkos_ENABLE_LARGE_MEM_TESTS``
      * Perform extra large memory tests
      * ``OFF``


Backend-specific options
------------------------
.. list-table::
    :widths: 25 65 35
    :header-rows: 1
    :align: left

    * -
      - Description/info
      - Default

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

    * * ``Kokkos_ENABLE_HIP_MULTIPLE_KERNEL_INSTANTIATIONS``
      * Instantiate multiple kernels at compile time - improve performance but increase compile time
      * ``OFF``

    * * ``Kokkos_ENABLE_HIP_RELOCATABLE_DEVICE_CODE``
      * Enable relocatable device code (RDC) for HIP
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

    * * ``Kokkos_ENABLE_HWLOC``
      * Whether to enable the HWLOC library
      * ``OFF``
    * * ``Kokkos_ENABLE_LIBDL``
      * Whether to enable the LIBDL library
      * ``ON``

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

    * * ``Kokkos_LIBDL_DIR`` or ``LIBDL_ROOT``
      * Location of LIBDL install prefix
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

Architectures
=============

CPU architectures
-----------------

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
      * Optimize for the ARMV8_THUNDERX2 architecture
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

    * * ``Kokkos_ARCH_KNC``
      * Optimize for KNC architecture
      * ``OFF``

    * * ``Kokkos_ARCH_KNL``
      * Optimize for KNL architecture
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


GPU Architectures
-----------------

NVIDIA GPUs
~~~~~~~~~~~

The Kokkos naming convention is to aggregate the eponym of the Nvidia GPU
microarchitecture and the associated CUDA Compute Capability.

``Kokkos_ARCH_<MICROARCHITECTURE><COMPUTE_CAPABILITY>``

If the CUDA backend is enabled and no NVIDIA GPU architecture is specified,
Kokkos will attempt to autodetect the architecture flag at configuration time.

.. list-table::
    :widths: 20 15 15 25 30
    :header-rows: 1
    :align: left

    * - **NVIDIA GPUs**
      - Architecture
      - Compute Capability
      - Models
      - Notes

    * * ``Kokkos_ARCH_HOPPER90``
      * Hopper
      * 9.0
      * H100
      * (since Kokkos 4.0)

    * * ``Kokkos_ARCH_ADA89``
      * Ada Lovelace
      * 8.9
      * L4, L40
      * (since Kokkos 4.1)

    * * ``Kokkos_ARCH_AMPERE86``
      * Ampere
      * 8.6
      * A40, A10, A16, A2
      *

    * * ``Kokkos_ARCH_AMPERE80``
      * Ampere
      * 8.0
      * A100, A30
      *

    * * ``Kokkos_ARCH_TURING75``
      * Turing
      * 7.5
      * T4
      *

    * * ``Kokkos_ARCH_VOLTA72``
      * Volta
      * 7.2
      *
      *

    * * ``Kokkos_ARCH_VOLTA70``
      * Volta
      * 7.0
      * V100
      *

    * * ``Kokkos_ARCH_PASCAL61``
      * Pascal
      * 6.1
      * P40, P4
      *

    * * ``Kokkos_ARCH_PASCAL60``
      * Pascal
      * 6.0
      * P100
      *

    * * ``Kokkos_ARCH_MAXWELL53``
      * Maxwell
      * 5.3
      *
      *

    * * ``Kokkos_ARCH_MAXWELL52``
      * Maxwell
      * 5.2
      * M60, M40
      *

    * * ``Kokkos_ARCH_MAXWELL50``
      * Maxwell
      * 5.0
      *
      *

    * * ``Kokkos_ARCH_KEPLER37``
      * Kepler
      * 3.7
      * K80
      *

    * * ``Kokkos_ARCH_KEPLER35``
      * Kepler
      * 3.5
      * K40, K20
      *

    * * ``Kokkos_ARCH_KEPLER32``
      * Kepler
      * 3.2
      *
      *

    * * ``Kokkos_ARCH_KEPLER30``
      * Kepler
      * 3.0
      * K10
      *


AMD GPUs
~~~~~~~~

.. list-table::
    :widths: 25 65 10
    :header-rows: 1
    :align: left

    * - **AMD GPUs**
      - Description/info
      - Default

    * * ``Kokkos_ARCH_AMD_GFX90A``
      * Optimize for AMD GPU MI200 series GFX90A :sup:`since Kokkos 4.2`
      * ``OFF``

    * * ``Kokkos_ARCH_AMD_GFX908``
      * Optimize for AMD GPU MI100 GFX908 :sup:`since Kokkos 4.2`
      * ``OFF``

    * * ``Kokkos_ARCH_AMD_GFX906``
      * Optimize for AMD GPU MI50/MI60 GFX906 :sup:`since Kokkos 4.2`
      * ``OFF``
    
    * * ``Kokkos_ARCH_AMD_GFX1100``
      * Optimize for AMD GPU 7900xt GFX1100 :sup:`since Kokkos 4.2` 
      * ``OFF``

    * * ``Kokkos_ARCH_AMD_GFX1030``
      * Optimize for AMD GPU V620/W6800 GFX1030 :sup:`since Kokkos 4.2` 
      * ``OFF``

    * * ``Kokkos_ARCH_VEGA900``
      * Optimize for AMD GPU MI25 GFX900 :sup:`removed in 4.0`
      * ``OFF``

    * * ``Kokkos_ARCH_VEGA906``
      * Optimize for AMD GPU MI50/MI60 GFX906 (Prefer ``Kokkos_ARCH_AMD_GFX906``)
      * ``OFF``

    * * ``Kokkos_ARCH_VEGA908``
      * Optimize for AMD GPU MI100 GFX908 (Prefer ``Kokkos_ARCH_AMD_GFX908``)
      * ``OFF``

    * * ``Kokkos_ARCH_VEGA90A``
      * Optimize for AMD GPU MI200 series GFX90A (Prefer ``Kokkos_ARCH_AMD_GFX90A``)
      * ``OFF``


.. list-table::
    :widths: 65 35
    :header-rows: 1
    :align: left

    * - AMD GPU
      - Kokkos ARCH
    
    * * ``7900xt``
      * AMD_GFX1100
      
    * * ``MI50/MI60``
      * AMD_GFX906
      
    * * ``MI100``
      * AMD_GFX908
      
    * * ``MI200`` series: ``MI210``, ``MI250``, ``MI250X``
      * AMD_GFX90A
    
    * * ``V620``
      * AMD_GFX1030
     
    * * ``W6800``
      * AMD_GFX1030

Intel GPUs
~~~~~~~~~~

.. list-table::
    :widths: 25 35 40
    :header-rows: 1
    :align: left

    * - CMake option
      - Architecture
      - Models

    * * ``Kokkos_ARCH_INTEL_PVC``
      * Xe-HPC(Ponte Vecchio)
      * Intel Data Center GPU Max 1550

    * * ``Kokkos_ARCH_INTEL_XEHP``
      * Xe-HP
      *

    * * ``Kokkos_ARCH_INTEL_DG1``
      * Iris XeMAX(DG1)
      *

    * * ``Kokkos_ARCH_INTEL_GEN12LP``
      * Gen12LP
      * Intel UHD Graphics 770

    * * ``Kokkos_ARCH_INTEL_GEN11``
      * Gen11
      * Intel UHD Graphics

    * * ``Kokkos_ARCH_INTEL_GEN9``
      * Gen9
      * Intel HD Graphics 510, Intel Iris Pro Graphics 580

    * * ``Kokkos_ARCH_INTEL_GEN``
      * Just-In-Time compilation*
      *

\* ``Kokkos_ARCH_INTEL_GEN`` enables Just-In-Time compilation for Intel GPUs
whereas all the other flags for Intel compilers request Ahead-Of-Time
compilation.

Just-In-Time (JIT) compilation means that the compiler is invoked again when
the binaries created are actually executed and only at that point the
architecture to compile for is determined.

On the other hand, Ahead-Of-Time (AOT) compilation describes the standard model
where the compiler is only invoked once to create the binary and the
architecture to compile for is determined before the program is run.
