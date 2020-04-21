# Chapter 4

# Compiling

This chapter explains how to compile Kokkos and how to link your application against Kokkos. Kokkos supports two build systems:

*  Using the embedded Makefile
*  Trilinos' CMake build system

Note that the two explicitly supported build methods should not be mixed. For example, do not include the embedded Makefile in your application build process, while explicitly linking against a pre-compiled Kokkos library in Trilinos. We also include specific advice for building for NVIDIA GPUs and for Intel Xeon Phi.

## 4.1 General Information

Kokkos consists mainly of header files. Only a few functions have to be compiled into object files outside of the application's source code. Those functions are contained in `.cpp` files inside the `kokkos/core/src` directory and its subdirectories. The files are internally protected with macros to prevent compilation if the related execution space is not enabled. Thus, it is not necessary to create a list of included object files specific to your compilation target; one may simply compile all `.cpp` files. The enabled features are controlled via macros which have to be provided in the compilation line or in the `KokkosCore_config.h` include file; a list of macros can be found in Table 4.1. In order to compile Kokkos, a C++11 compliant compiler is needed. For an up to date list of compilers which are tested on a nightly basis, please refer to the README on the github repository. At the time of writing supported compilers include:

    Primary tested compilers on X86
        GCC 4.8.4, 4.9.3, 5.1.0, 5.3.0, 6.1.0;
        Intel 15.0.2, 16.0.1, 16.0.3, 17.0.098, 17.1.132;  
        Clang 3.6.1, 3.7.1, 3.8.1, 3.9.0;  
        Cuda 7.0, 7.5, 8.0;
        PGI 17.10  
    Primary tested compilers on Power 8
        XL 13.1.3 (OpenMP, Serial)
        GCC 5.4.0 (OpenMP, Serial)
        Cuda 8.0, 9.0 (with gcc 5.4.0);
    Primary tested compilers on Intel KNL
        GCC 6.2.0
        Intel 16.4.258 (with gcc 4.7.4)
        Intel 17.2.174 (with gcc 4.9.3)
        Intel 18.0.128 (with gcc 4.9.3)
    
    Secondary tested compilers
        CUDA 7.0, 7.5 (with gcc 4.8.4)
        CUDA 8.0 (with gcc 5.3.0 on X86)
        CUDA 8.0 (with gcc 5.4.0 on Power8)
        Clang 4.0 (with CUDA 8.0, using Clang as the CUDA compiler, requires 384.x CUDA drivers)
    
    Other working compilers
        Cygwin 2.1.0 64bit (with gcc 4.9.3 on X86)

    Known non-working combinations
        Pthreads backend (on Power 8)


<h4>Table 4.1: Configuration Macros</h4>
  
 Macro | Effect | Comment
 :--- |:--- |:---
`KOKKOS_ENABLE_CUDA`| Enable the CUDA execution space. |Requires a compiler capable of understanding CUDA-C. See Section 4.4.
`KOKKOS_ENABLE_OPENMP`| Enable the OpenMP execution space. |Requires the compiler to support OpenMP (e.g., `-fopenmp`).
`KOKKOS_ENABLE_PTHREADS`| Enable the Threads execution space. | Requires linking with `libpthread`.
`KOKKOS_ENABLE_Serial`| Enable the Serial execution space. |
`KOKKOS_ENABLE_CXX11`| Enable internal usage of C++11 features. | The code needs to be compiled with the C++11 standard. Most compilers accept the -std=c++11 flag for this.
`KOKKOS_ENABLE_HWLOC`| Enable thread and memory pinning via hwloc. | Requires linking with `libhwloc`. 


## 4.2 Using Kokkos' Makefile system

The base of the build system is the file `Makefile.kokkos`; it is designed to be included by application Makefiles. It contains logic to (re)generate the `KokkosCore_config.h` file if necessary, build the Kokkos library, and provide updated compiler and linker flags. 

The system can digest a number of variables which are used to configure Kokkos settings and then parses the variables for Keywords. This allows for multiple options to be given for each variable. The separator doesn't matter as long as it doesn't interact with the Make system. A list of variables, their meaning and options are given in Table 4.2.

A word of caution on where to include the `Makefile.kokkos`: since the embedded Makefiles define targets, it is usually better to include it after the first application target has been defined. Since that target can't use the flags from the embedded Makefiles, it should be a meta target:

    CXX=g++
    default: main
    include Makefile.kokkos
    main: $(KOKKOS_LINK_DEPENDS) $(KOKKOS_CPP_DEPENDS) main.cpp
          $(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) \
          $(KOKKOS_LDFLAGS) $(KOKKOS_LIBS) main.cpp -o main

More example application Makefiles can be found in the tutorial examples under `kokkos/example/tutorial`.

Kokkos provides a script `generate_makefile.bash` which can generate a Makefile for building and installing the library as well as building and running the tests. Please run `generate_makefile.bash --help` to see options. Note that paths given to the script must be absolute paths, and the script must be run with the `bash` shell (the script will do this if it is run directly, i.e., as `./generate_makefile.bash`).

<h4>Table 4.2: Variables for the Embedded Makefile</h4>

Variable  | Description
 ---: |:---
`KOKKOS_PATH (IN)` | Path to the Kokkos root or install directory. One can either build against an existing install of Kokkos or use its source directly for an embedded build. In the former case the "Input variables" are set inside the embedded Makefile.kokkos and it is not valid to set them differently in the including Makefile. 
`CUDA_PATH (IN)` | Path to the Cuda toolkit root directory. 
`KOKKOS_DEVICES (IN)` | Execution and Memory Spaces that should be enabled.
`Options`<br>`    Default` | OpenMP, Serial, Pthreads, Cuda <br> OpenMP 
`KOKKOS_ARCH (IN)` | The backend architecture to build for.
`Options` <br><br><br> `Default` | KNL, KNC, SNB, HSW, BDW, SKX, Kepler, Kepler30, Kepler35, Kepler37, Maxwell, Maxwell50, Pascal60, Pascal61, Volta70, Turing75, ARMv8, ARMv81, ARMv8-ThunderX, BGQ, Power7, Power8, Power9 <br><br> (no particular architecture flags are set).
`KOKKOS_USE_TPLS (IN)` | Enable optional third party libraries.
`Options` <br> `Default`  | hwloc, librt, experimental_memkind <br> (none)
`KOKKOS_OPTIONS (IN)` | Enable optional settings
`Options` <br> `Default` | aggressive_vectorization <br> (none)
`KOKKOS_CUDA_OPTIONS (IN)` | Enable optional settings specific to CUDA.
`Options` <br> `Default` | force_uvm, use_ldg, rdc, enable_lambda <br> (none)
`HWLOC_PATH (IN)` | Path to the hardware locality library if enabled.
`KOKKOS_DEBUG (IN)` | Enable debugging.
`Options` <br> `Default` | yes, no <br> no
`KOKKOS_CXX_STANDARD (IN)` | Set the C++ standard to be used.
`Options` <br> `Default`  | C++11 <br> C++11
`KOKKOS_CPPFLAGS (OUT)` | Preprocessor flags (include directories and defines). Add this to applications compiler and preprocessor flags.
`KOKKOS_CXXFLAGS (OUT)` | Compiler flags. Add this to the applications compiler flags.
`KOKKOS_LDFLAGS (OUT)` | Linker flags. Add this to the applications linker flags.
`KOKKOS LIBS (OUT)` | Libraries required by Kokkos. Add this to the link line after the linker flags.
`KOKKOS_CPP_DEPENDS (OUT)` |  Dependencies for compilation units which include any Kokkos header files. <br> Add this as a dependency to compilation targets including any Kokkos code.
`KOKKOS_LINK_DEPENDS (OUT)` | Dependencies of an application linking in the Kokkos library. Add this to the dependency list of link targets.
`CXXFLAGS (IN)` | User provided compiler flags which will be used to compile the Kokkos library.
`CXX (IN)` | The compiler used to compile the Kokkos library.


## 4.3 Using Trilinos' CMake build system

The Trilinos project (see `trilinos.org`) is an effort to develop algorithms and enabling technologies within an object-oriented software framework for the solution of large-scale, complex multiphysics engineering and scientific problems. Trilinos is organized into packages. Even though Kokkos is a stand-alone software project, Trilinos uses Kokkos extensively. Thus, Trilinos' source code includes Kokkos' source code, and builds Kokkos as part of its build process.

Trilinos' build system uses CMake. Thus, in order to build Kokkos as part of Trilinos, you must first install CMake (version `2.8.12` or newer; CMake `3.x` works). To enable Kokkos when building Trilinos, set the CMake option `Trilinos_ENABLE_Kokkos`. Trilinos' build system lets packages express dependencies on other packages or external libraries. If you enable any Trilinos package (e.g., Tpetra) that has a required dependency on Kokkos, Trilinos will enable Kokkos automatically. Configuration macros are automatically inferred from Trilinos settings. For example, if the CMake option `Trilinos_ENABLE_OpenMP` is `ON`, Trilinos will define the macro `KOKKOS_ENABLE_OPENMP`. Trilinos' build system will autogenerate the previously mentioned `KokkosCore_config.h` file that contains those macros.

Trilinos' CMake build system utilizes Kokkos' build system to set compiler flags, compiler options, architectures, etc. CMake variables `CMAKE_CXX_COMPILER`, `CMAKE_C_COMPILER`, and `CMAKE_FORTRAN_COMPILER` are used to specify the compiler. To configure Trilinos for various archictures, with Kokkos enabled, the CMake variable `KOKKOS_ARCH` should be set to the appropriate architecture as specified in the Table 4.3.

<h4>Table 4.3: Architecture Variables</h4>

Variable  | Description
 ---: |:---
`AMDAVX` | AMD CPU
`ARMv80` | ARMv8.0 Compatible CPU
`ARMv81` | ARMv8.1 Compatible CPU
`ARMv8-ThunderX` | ARMv8 Cavium ThunderX CPU
`BGQ` | IBM Blue Gene Q
`Power7` | IBM POWER7 and POWER7+ CPUs
`Power8` | IBM POWER8 CPUs
`Power9` | IBM POWER9 CPUs
`WSM` | Intel Westmere CPUs
`SNB` | Intel Sandy/Ivy Bridge CPUs
`HSW` | Intel Haswell CPUs
`BDW` | Intel Broadwell Xeon E-class CPUs
`SKX` | Intel Sky Lake Xeon E-class HPC CPUs (AVX512)
`KNC` | Intel Knights Corner Xeon Phi
`KNL` | Intel Knights Landing Xeon Phi
`Kepler30` | NVIDIA Kepler generation CC 3.0
`Kepler32` | NVIDIA Kepler generation CC 3.2
`Kepler35` | NVIDIA Kepler generation CC 3.5
`Kepler37` | NVIDIA Kepler generation CC 3.7
`Maxwell50` | NVIDIA Maxwell generation CC 5.0
`Maxwell52` | NVIDIA Maxwell generation CC 5.2
`Maxwell53` | NVIDIA Maxwell generation CC 5.3
`Pascal60` | NVIDIA Pascal generation CC 6.0
`Pascal61` | NVIDIA Pascal generation CC 6.1
`Volta70` | NVIDIA Volta generation CC 7.0
`Volta72` | NVIDIA Volta generation CC 7.2


Multiple architectures can be specified by separting the architecture variables with a semi-colon, for example `KOKKOS_ARCH:STRING="HSW;Kepler35` sets architecture variables for a machine with Intel Haswell CPUs and a NVIDIA Tesla K40 GPU. In addition, when setting the `KOKKOS_ARCH` variable it is not necessary to pass required architecture-specific flags to CMake, for example via the `CMAKE_CXX_FLAGS` variable.

Several Trilinos packages with CUDA support currently require the use of UVM (note UVM is enabled by default when configuring Trilinos with CUDA enabled, unless the user explictly disables it). To ensure proper compilation and execution for such packages, the environment variables `export CUDA_LAUNCH_BLOCKING=1` and `export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1` must be set.

We refer readers to Trilinos' documentation for further details.

## 4.4 Building for CUDA

Any Kokkos application compiled for CUDA embeds CUDA code via template metaprogramming. Thus, the whole application must be built with a CUDA-capable compiler. (At the moment, the only such compilers are NVIDIA's NVCC and Clang 4.0.) More precisely, every compilation unit containing a Kokkos kernel or a function called from a Kokkos kernel has to be compiled with a CUDA-capable compiler. This includes files containing `Kokkos::View` allocations which call an initialization kernel.

All current versions of the NVCC compiler have shortcomings when used as the main compiler for a project, in particular when part of a complex build system. For example, it does not understand most GCC command-line options, which must be prepended by the `-Xcompiler` flag when calling NVCC. Kokkos comes with a shell script, called `nvcc_wrapper`, that wraps NVCC to address these issues. We intend this as a drop-in replacement for a normal GCC-compatible compiler (e.g., GCC or Intel) in your build system. It analyzes the provided command-line options and prepends them correctly. It also adds the correct flags for compiling generic C++ files containing CUDA code (e.g., `*.cpp, *.cxx,` or `*.CC`). By default `nvcc_wrapper` calls `g++` as the host compiler. You may override this by providing NVCC's `-ccbin` option as a compiler flag. The default can be set by editing the script itself or by setting the environment variable `NVCC_WRAPPER_DEFAULT_COMPILER`.

Many people use a system like Environment Modules (see `http://modules.sourceforge.net/`) to manage their shell environment. When using a module system, it can be useful to provide different versions for different back-end compiler types (e.g., `icpc, pgc++, g++,` and `clang`). To use the `nvcc_wrapper` in conjunction with MPI wrappers, simply overwrite which C++ compiler is called by the MPI wrapper. For example, you can reset OpenMPI's C++ compiler by setting the `OMPI_CXX` environment variable. Make sure that `nvcc_wrapper` calls the host compiler with which the MPI library was compiled.

## 4.5 Execution Space Restrictions

Currently, Kokkos organizes its execution spaces into three categories:

 - Host Serial: A top-level `Serial` execution space with no parallelism or concurrency
 - Host Parallel: Typically a threading model for CPUs, currently: `OpenMP`, `Threads`, and `QThreads`.
 - Device Parallel: Typically an attached GPU, currently: `CUDA`, `OpenMPTarget`, and `ROCm`.

The current Kokkos policy is to allow users, at compile time, to enable *at most one* execution space from each category. This prevents incompatibilities between different spaces in the same category from degrading the user's correctness and performance.

**[[Chapter 5: Initialization|Initialization]]**
