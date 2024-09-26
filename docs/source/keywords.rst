.. include:: mydefs.rst

CMake Keywords
##############

.. important::

   With version 3.0 all Kokkos CMake keywords are prefixed with `Kokkos_` which is case-sensitive.

   Recall that to set a keyword in CMake you used the syntax ``-Dkeyword_name=value``.

.. note::
   You may use the ``ccmake`` curses GUI to browse the complete list of
   available CMake options and their current settings. It may be more current to your
   Kokkos version than this Wiki's list.

This page is organized in four sections:

- :ref:`keywords_backends`
- :ref:`keywords_enable_options`
- :ref:`keywords_tpls`
- :ref:`keywords_arch`

.. _keywords_backends:

Backend selection
=================

**Default State:**
All backends are disabled by default.  This ensures you explicitly choose the
backends you need for your specific hardware setup.
If no backend is enabled explicitly, the Serial backend will be enabled.

**Enabling Backends:**
You can enable backends by configuring with ``-DKokkos_ENABLE_<BACKEND>=ON``
flag, where ``<BACKEND>`` is replaced with the specific backend you want to
enable (e.g. ``-DKokkos_ENABLE_CUDA=ON`` for CUDA).

**Restrictions:**
  Mutual Exclusion: You can only have one device backend (e.g., CUDA, HIP,
  SYCL) and one host parallel backend (e.g., OpenMP, C++ threads) enabled at
  the same time. This is because these backends manage parallelism in
  potentially conflicting ways.

  Host Backend Requirement: At least one host backend must always be enabled.
  This is because Kokkos code execution typically starts on the host (CPU)
  before potentially being offloaded to devices (GPUs, accelerators). If you
  don't explicitly enable a host backend, Kokkos will automatically enable the
  Serial backend, which provides a sequential execution model.

Serial backend
--------------

.. list-table::
    :widths: 25 65
    :header-rows: 1
    :align: left

    * -
      - Description/info

    * - ``Kokkos_ENABLE_SERIAL``
      - To build the Serial backend targeting CPUs

Host parallel backends
----------------------

.. list-table::
    :widths: 25 65
    :header-rows: 1
    :align: left

    * -
      - Description/info

    * - ``Kokkos_ENABLE_OPENMP``
      - To build the OpenMP backend targeting CPUs

    * - ``Kokkos_ENABLE_THREADS``
      - To build the C++ Threads backend

    * - ``Kokkos_ENABLE_HPX``
      - :red:`[Experimental]` To build the HPX backend

Device backends
---------------

.. list-table::
    :widths: 25 65
    :header-rows: 1
    :align: left

    * -
      - Description/info

    * - ``Kokkos_ENABLE_CUDA``
      - To build the CUDA backend targeting NVIDIA GPUs

    * - ``Kokkos_ENABLE_HIP``
      - To build the HIP backend targeting AMD GPUs

    * - ``Kokkos_ENABLE_SYCL``
      - :red:`[Experimental]` To build the SYCL backend targeting Intel GPUs

    * - ``Kokkos_ENABLE_OPENMPTARGET``
      - :red:`[Experimental]` To build the OpenMP target backend


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

    * * ``Kokkos_ENABLE_DEPRECATED_CODE_3``
      * Enable deprecated code in the Kokkos 3.x series :red:`[Removed in 4.3]`
      * ``OFF``

    * * ``Kokkos_ENABLE_DEPRECATED_CODE_4``
      * Enable deprecated code in the Kokkos 4.x series
      * ``ON``

    * * ``Kokkos_ENABLE_DEPRECATION_WARNINGS``
      * Whether to raise warnings at compile time when using deprecated Kokkos facilities
      * ``ON``

    * * ``Kokkos_ENABLE_TUNING``
      * Create bindings for tuning tools
      * ``OFF``

    * * ``Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION``
      * Aggressively vectorize loops
      * ``OFF``

Debugging
---------
.. list-table::
    :widths: 25 65 35
    :header-rows: 1
    :align: left

    * -
      - Description/info
      - Default

    * * ``Kokkos_ENABLE_DEBUG``
      * Activate extra debug features - may increase compile times
      * ``OFF``

    * * ``Kokkos_ENABLE_DEBUG_BOUNDS_CHECK``
      * Use bounds checking - will increase runtime
      * ``OFF``

    * * ``Kokkos_ENABLE_DEBUG_DUALVIEW_MODIFY_CHECK``
      * Debug check on dual views
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

    * * ``Kokkos_ENABLE_ATOMICS_BYPASS``
      * Disable atomics when no host parallel nor device backend is enabled for Serial only builds (since Kokkos 4.3)
      * ``OFF``
    * * ``Kokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC``
      * Enable ``cudaMallocAsync`` (requires CUDA Toolkit version 11.2 or higher). This optimization may
      improve performance in applications with multiple CUDA streams per device, but it
      is known to be incompatible with MPI distributions built on older versions of UCX
      and many Cray MPICH instances. See `kokkos/kokkos#7353 <https://github.com/kokkos/kokkos/pull/7353>`_
      * ``ON``
	
    * * ``Kokkos_ENABLE_IMPL_HPX_ASYNC_DISPATCH``
      * Enable asynchronous dispatch for the HPX backend
      * ``ON``
	
``Kokkos_ENABLE_CUDA_LAMBDA`` default value is ``OFF`` until 3.7 and ``ON`` since 4.0

``Kokkos_ENABLE_IMPL_CUDA_MALLOC_ASYNC`` default value is ``OFF`` until 4.2, and ``ON`` until 4.5

Any CMake keyword containing `IMPL` can fundamentally alter the underlying implementation
of Kokkos on a given backend. It is encouraged not to set these unless the user knows what
they are doing can accept the behavior changing without warning.


Development
-----------
These are intended for developers of Kokkos.  If you are a user, you probably
should not be setting these.

.. list-table::
    :widths: 25 65 35
    :header-rows: 1
    :align: left

    * -
      - Description/info
      - Default

    * * ``Kokkos_ENABLE_COMPILER_WARNINGS``
      * Print all compiler warnings
      * ``OFF``

    * * ``Kokkos_ENABLE_HEADER_SELF_CONTAINMENT_TESTS``
      * Check that headers are self-contained
      * ``OFF``

    * * ``Kokkos_ENABLE_LARGE_MEM_TESTS``
      * Perform extra large memory tests
      * ``OFF``

.. _keywords_tpls:

Third-party Libraries (TPLs)
============================

The following options control enabling TPLs:

.. list-table::
    :widths: 30 40 10 20
    :header-rows: 1
    :align: left

    * -
      - Description/info
      - Default
      - Notes

    * * ``Kokkos_ENABLE_HWLOC``
      * Whether to enable the HWLOC library
      * ``OFF``
      *
    * * ``Kokkos_ENABLE_LIBDL``
      * Whether to enable the LIBDL library
      * ``ON``
      *
    * * ``Kokkos_ENABLE_ONEDPL``
      * Whether to enable the oneDPL library when using the SYCL backend
      * ``ON``
      *
    * * ``Kokkos_ENABLE_ROCTHRUST``
      * Whether to enable the rocThrust library when using the HIP backend
      * ``ON``
      * (since Kokkos 4.3)

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

    * * ``Kokkos_ARCH_HSW``
      * Optimize for HSW architecture
      * ``OFF``

    * * ``Kokkos_ARCH_KNC``
      * Optimize for KNC architecture
      * ``OFF``

    * * ``Kokkos_ARCH_KNL``
      * Optimize for KNL architecture
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

The Kokkos naming convention is to aggregate AMD\_ and the architecture flag.

``Kokkos_ARCH_AMD_<ARCHITECTURE_FLAG>``

If the HIP backend is enabled and no AMD GPU architecture is specified,
Kokkos will attempt to autodetect the architecture flag at configuration time.

.. list-table::
    :widths: 30 15 25 30
    :header-rows: 1
    :align: left

    * - **AMD GPUs**
      - Architecture flags
      - Models
      - Notes

    * * ``Kokkos_ARCH_AMD_GFX942``
      * GFX942
      * MI300A, MI300X
      * (since Kokkos 4.2)

    * * ``Kokkos_ARCH_AMD_GFX940``
      * GFX940
      * MI300A (pre-production)
      * (since Kokkos 4.2.1)

    * * ``Kokkos_ARCH_AMD_GFX90A``
      * GFX90A
      * MI200 series 
      * (since Kokkos 4.2)

    * * ``Kokkos_ARCH_AMD_GFX908``
      * GFX90A
      * MI100
      * (since Kokkos 4.2)

    * * ``Kokkos_ARCH_AMD_GFX906``
      * GFX906
      * MI50, MI60
      * (since Kokkos 4.2)
    
    * * ``Kokkos_ARCH_AMD_GFX1100``
      * GFX1100
      * 7900xt
      * (since Kokkos 4.2)

    * * ``Kokkos_ARCH_AMD_GFX1030``
      * GFX1030
      * V620, W6800
      * (since Kokkos 4.2)

    * * ``Kokkos_ARCH_VEGA90A``
      * GFX90A
      * MI200 series
      * Prefer ``Kokkos_ARCH_AMD_GFX90A``

    * * ``Kokkos_ARCH_VEGA908``
      * GFX908
      * MI100 
      * Prefer ``Kokkos_ARCH_AMD_GFX908``

    * * ``Kokkos_ARCH_VEGA906``
      * GFX906
      * MI50, MI60 
      * Prefer ``Kokkos_ARCH_AMD_GFX906``

    * * ``Kokkos_ARCH_VEGA900``
      * GFX900
      * MI25 
      * removed in 4.0

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
      * Xe-HPC (Ponte Vecchio)
      * Intel Data Center GPU Max 1550

    * * ``Kokkos_ARCH_INTEL_XEHP``
      * Xe-HP
      *

    * * ``Kokkos_ARCH_INTEL_DG1``
      * Iris Xe MAX (DG1)
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

    * *
      *
      *

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
