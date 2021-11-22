# Chapter 4

# Compiling

This chapter explains how to compile Kokkos and how to link your application against Kokkos. Kokkos supports three methods to build:

*  General CMake build system
*  Trilinos' CMake build system
*  Embedded GNU Makefile

Note that the build methods listed above should not be mixed. For example, do not include the GNU Makefile in your application build process, while explicitly linking against a pre-compiled Kokkos library in Trilinos. We also include specific advice for building for NVIDIA GPUs and Intel Xeon Phi.

## 4.1 General Information

Kokkos consists mainly of header files. Only a few functions have to be compiled into object files outside of the application's source code. Those functions are contained in `.cpp` files inside the `kokkos/core/src` directory and its subdirectories. The files are internally protected with macros to prevent compilation if the related execution space is not enabled. Thus, it is not necessary to create a list of included object files specific to your compilation target; one may simply compile all `.cpp` files. The enabled features are controlled via macros which have to be provided in the compilation line or the generated `KokkosCore_config.h` include file; a subset of the macros can be found in Table 4.1.  For the most part, all of these macros are enabled/disabled using the options and settings controlled through one of the build methods previously mentioned.

To compile Kokkos, a C++14 compliant compiler is needed. For an up to date list of compilers that are tested on a nightly basis, please refer to the README on the GitHub repository. At the time of writing supported compilers include:

```
Minimum Compiler Versions

    GCC: 5.3.0
    Clang: 4.0.0
    Intel: 17.0.1
    NVCC: 9.2.88
    NVC++: 21.5
    ROCM: 4.3
    MSVC: 19.29
    IBM XL: 16.1.1
    Fujitsu: 4.5.0
    ARM/Clang 20.1

Primary Tested Compilers

    GCC: 5.3.0, 6.1.0, 7.3.0, 8.3, 9.2, 10.0
    NVCC: 9.2.88, 10.1, 11.0
    Clang: 8.0.0, 9.0.0, 10.0.0, 12.0.0
    Intel 17.4, 18.1, 19.5
    MSVC: 19.29
    ARM/Clang: 20.1
    IBM XL: 16.1.1
    ROCM: 4.3.0

Build system:

    CMake >= 3.16: required
    CMake >= 3.18: Fortran linkage. This does not affect most mixed Fortran/Kokkos builds. See build issues.
    CMake >= 3.21.1 for NVC++
    
```
<h4>Table 4.1: Configuration Macros (KokkosCore_config.h)</h4>
  
 Macro | Effect | Comment
 :--- |:--- |:---
`KOKKOS_ENABLE_CUDA`| Enable the CUDA execution space. |Requires a compiler capable of understanding CUDA-C. See Section 4.4.
`KOKKOS_ENABLE_OPENMP`| Enable the OpenMP execution space. |Requires the compiler to support OpenMP (e.g., `-fopenmp`).
`KOKKOS_ENABLE_PTHREADS`| Enable the Threads execution space. | Requires linking with `libpthread`.
`KOKKOS_ENABLE_SERIAL`| Enable the Serial execution space. |
`KOKKOS_ENABLE_CXX11`| Enable internal usage of C++11 features. | The code needs to be compiled with the C++11 standard. Most compilers accept the -std=c++11 flag for this.
`KOKKOS_ENABLE_HWLOC`| Enable thread and memory pinning via hwloc. | Requires linking with `libhwloc`. 

## 4.2 Using General CMake build system

### Installing and Using Kokkos

Kokkos provides a CMake style build system.
As C++ continues to develop for C++20 and beyond, CMake is likely to provide the most robust support
for C++.  Applications heavily leveraging Kokkos are strongly encouraged to use a CMake build system.  Kokkos requires CMake version 3.10 and above.

You can either use Kokkos as an installed package (encouraged) or use Kokkos in-tree included in your project.

### Using Kokkos installed Package
With the Kokkos package installed, you build and link with the Kokkos library using CMake by adding the following to you your `CMakeLists.txt`:
````cmake
find_package(Kokkos REQUIRED)
````
Then for every executable or library in your project:
````cmake
target_link_libraries(myTarget Kokkos::kokkos)
````
The target_link_libraries command will find and include all of the necessary pre-processor, compiler, and linker flags that are required for an application using Kokkos.  When running CMake for your project you will need to specify the directory containing the Kokkos package:  
````
-DKokkos_ROOT=<Kokkos Install Directory>/lib64/cmake/Kokkos
````
If compiling with something other than g++, your application should use a compiler that is consistent with that used to build the Kokkos package.  This is especially true when using nvcc_wrapper.
````
-DCMAKE_CXX_COMPILER=<Kokkos Install Directory>/bin/nvcc_wrapper
````

**Important note** 
With Kokkos release 3.0 the externally defined CMAKE_CXX_FLAGS are not propagated to projects that include the kokkos package.  This limitation is especially important when using Clang compilers with gcc and Cuda.  The Clang options that are provided via the CMAKE_CXX_FLAGS with the Kokkos project are illustrated below.

```
--gcc-toolchain=<path to gcc source tree>
--cuda-path=<path to cuda source>
``` 

### Using Kokkos in-tree build
If building in-tree, the Kokkos source directory must be within a sub-directory of your application source tree (relative to the location of your application CMakeLists.txt)

To include Kokkos in the application add the following to CMakeLists.txt:
````cmake
add_subdirectory(<path to Kokkos dir relative to your CMakeList.txt>)
include_directories(${Kokkos_INCLUDE_DIRS_RET})
target_link_libraries(myTarget kokkos)
```` 
The include_directories command is necessary to update the application include paths, and the target link libraries command links your executable to the Kokkos library.  It does not require a package name.
Using this method, the Kokkos options necessary to specify the devices, arch and options must be specified with your application CMake command.  See below for the list of available settings (keywords) 


## Configuring Kokkos with CMake
A very basic installation is done with:

````bash
> mkdir build
> cd build
> cmake ${srcdir} \
 -DCMAKE_CXX_COMPILER=g++ \
 -DCMAKE_INSTALL_PREFIX=${my_install_folder} \
````
which builds and installed a default Kokkos when you run `make install`.
There are numerous device backends, options, and architecture-specific optimizations that can be configured, e.g.
````bash
> cmake ${srcdir} \
 -DCMAKE_CXX_COMPILER=g++ \
 -DCMAKE_INSTALL_PREFIX=${my_install_folder} \
 -DKokkos_ENABLE_OPENMP=On
````
which activates the OpenMP backend. All of the options controlling device backends, options, architectures, and third-party libraries (TPLs) are given below under the keywords listing.

### Using generate_makefile.bash
As an alternative to calling the cmake command directly, the generate_makefile.bash command can be used to configure the CMake build environment.  The generate_makefile.bash equivalent to the above OpenMP example is as follows:

````bash
> ${srcdir}/generate_makefile.bash --compiler=g++ \
  --with-openmp --prefix=${my_install_folder}
````
For a full list of generate_makefile.bash options use the command 
````bash
> ${srcdir}/generate_makefile.bash --help
````

### Spack
An alternative to manually building with CMake is to use the Spack package manager.
To do so, download [Spack](https://github.com/spack/spack) and add it to your path by sourcing the appropriate env file in the share folder, e.g.
````bash
> source spack/share/spack/setup-env.sh 
````
A basic installation would be done as:
````bash
> spack install kokkos
````
Spack allows options and compilers to be tuned in the install command.
````bash
> spack install kokkos@3.0 %gcc@7.3.0 +openmp
````
This example illustrates the three most common parameters to Spack:
* Variants: specified with, e.g. `+openmp`, this activates (or deactivates with, e.g. `~openmp`) certain options.
* Version:  immediately following `kokkos` the `@version` can specify a particular Kokkos to build
* Compiler: a default compiler will be chosen if not specified, but an exact compiler version can be given with the `%`option.

For a complete list of Kokkos options, run:
````bash
> spack info kokkos
````

#### Spack Development
Spack currently installs packages to a location determined by a unique hash. This hash name is not really "human readable".
Generally, Spack usage should never really require you to reference the computer-generated unique install folder.
If you must know, you can locate Spack Kokkos installations with:
````bash
> spack find -p kokkos ...
````
where `...` is the unique spec identifying the particular Kokkos configuration and version.

A better way to use Spack for doing Kokkos development is the DIY feature of Spack.
If you wish to develop Kokkos itself, go to the Kokkos source folder:
````bash
> spack diy -u cmake kokkos@diy ...
````
where `...` is a Spack spec identifying the exact Kokkos configuration.
This then creates a `spack-build` directory where you can run `make`.

If doing development on a downstream project, you can do almost the same thing.
````bash
> spack diy -u cmake ${myproject}@${myversion} ... ^kokkos...
````
where the `...` are the specs for your project and the desired Kokkos configuration.
Again, a `spack-build` directory will be created where you can run `make`.

Spack has a few idiosyncrasies that make building outside of Spack annoying related to Spack forcing the use of a compiler wrapper. This can be worked around by having a `-DSpack_WORKAROUND=On` given your CMake. Then add the block of code to your CMakeLists.txt:

````cmake
if (Spack_WORKAROUND)
 set(SPACK_CXX $ENV{SPACK_CXX})
 if(SPACK_CXX)
   set(CMAKE_CXX_COMPILER ${SPACK_CXX} CACHE STRING "the C++ compiler" FORCE)
   set(ENV{CXX} ${SPACK_CXX})
 endif()
endif()
````

### Kokkos CMake Keyword Listing
Note that with version 3.0 all Kokkos CMake keywords are prefixed with `Kokkos_` which is case sensitive.  
#### Device Backends
Device backends can be enabled by specifying `-DKokkos_ENABLE_X`.

* Kokkos_ENABLE_CUDA
    * Whether to build CUDA backend
    * BOOL Default: OFF
* Kokkos_ENABLE_HIP
    * Whether to build HIP backend
    * BOOL Default: OFF
* Kokkos_ENABLE_OPENMPTARGET
    * Whether to build OPENMPTARGET backend (experimental)
    * BOOL Default: OFF
* Kokkos_ENABLE_SYCL
    * Whether to build SYCL backend (experimental)
    * BOOL Default: OFF
* Kokkos_ENABLE_HPX
    * Whether to build HPX backend (experimental)
    * BOOL Default: OFF
* Kokkos_ENABLE_OPENMP
    * Whether to build OpenMP backend
    * BOOL Default: OFF
* Kokkos_ENABLE_PTHREAD
    * Whether to build Pthread backend
    * BOOL Default: OFF
* Kokkos_ENABLE_SERIAL
    * Whether to build serial  backend
    * BOOL Default: ON

#### Enable Options
Options can be enabled by specifying `-DKokkos_ENABLE_X`.

* Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION
    * Whether to aggressively vectorize loops
    * BOOL Default: OFF
* Kokkos_ENABLE_COMPILER_WARNINGS
    * Whether to print all compiler warnings
    * BOOL Default: OFF
* Kokkos_ENABLE_CUDA_CONSTEXPR
    * Whether to activate experimental relaxed constexpr functions
    * BOOL Default: OFF
* Kokkos_ENABLE_CUDA_LAMBDA
    * Whether to activate experimental lambda features
    * BOOL Default: OFF
* Kokkos_ENABLE_CUDA_LDG_INTRINSIC
    * Whether to use CUDA LDG intrinsics
    * BOOL Default: OFF
* Kokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE
    * Whether to enable relocatable device code (RDC) for CUDA
    * BOOL Default: OFF
* Kokkos_ENABLE_CUDA_UVM
    * Whether to use unified memory (UM) by default for CUDA
    * BOOL Default: OFF
* Kokkos_ENABLE_DEBUG
    * Whether to activate extra debug features - may increase compile times
    * BOOL Default: OFF
* Kokkos_ENABLE_DEBUG_BOUNDS_CHECK
    * Whether to use bounds checking - will increase runtime
    * BOOL Default: OFF
* Kokkos_ENABLE_DEBUG_DUALVIEW_MODIFY_CHECK
    * Debug check on dual views
    * BOOL Default: OFF
* Kokkos_ENABLE_DEPRECATED_CODE
    * Whether to enable deprecated code
    * BOOL Default: OFF
* Kokkos_ENABLE_EXAMPLES
    * Whether to enable building examples
    * BOOL Default: OFF
* Kokkos_ENABLE_HPX_ASYNC_DISPATCH
    * Whether HPX supports asynchronous dispatch
    * BOOL Default: OFF
* Kokkos_ENABLE_LARGE_MEM_TESTS
    * Whether to perform extra large memory tests
    * BOOL_Default: OFF
* Kokkos_ENABLE_PROFILING
    * Whether to create bindings for profiling tools
    * BOOL Default: ON
* Kokkos_ENABLE_PROFILING_LOAD_PRINT
    * Whether to print information about which profiling tools got loaded
    * BOOL Default: OFF
* Kokkos_ENABLE_TESTS
    * Whether to build serial  backend
    * BOOL Default: OFF

#### Other Options
* Kokkos_CXX_STANDARD
    * The C++ standard for Kokkos to use: c++11, c++14, c++17, or c++20. This should be given in CMake style as 11, 14, 17, or 20.
    * STRING Default: 11

#### Third-party Libraries (TPLs)
The following options control enabling TPLs:
* Kokkos_ENABLE_HPX
    * Whether to enable the HPX library
    * BOOL Default: OFF
* Kokkos_ENABLE_HWLOC
    * Whether to enable the HWLOC library
    * BOOL Default: Off
* Kokkos_ENABLE_LIBNUMA
    * Whether to enable the LIBNUMA library
    * BOOL Default: Off
* Kokkos_ENABLE_MEMKIND
    * Whether to enable the MEMKIND library
    * BOOL Default: Off
* Kokkos_ENABLE_LIBDL
    * Whether to enable the LIBDL library
    * BOOL Default: On
* Kokkos_ENABLE_LIBRT
    * Whether to enable the LIBRT library
    * BOOL Default: Off

The following options control finding and configuring non-CMake TPLs:
* Kokkos_CUDA_DIR or CUDA_ROOT
    * Location of CUDA install prefix for libraries
    * PATH Default:
* Kokkos_HWLOC_DIR or HWLOC_ROOT
    * Location of HWLOC install prefix
    * PATH Default:
* Kokkos_LIBNUMA_DIR or LIBNUMA_ROOT
    * Location of LIBNUMA install prefix
    * PATH Default:
* Kokkos_MEMKIND_DIR or MEMKIND_ROOT
    * Location of MEMKIND install prefix
    * PATH Default:
* Kokkos_LIBDL_DIR or LIBDL_ROOT
    * Location of LIBDL install prefix
    * PATH Default:
* Kokkos_LIBRT_DIR or LIBRT_ROOT
    * Location of LIBRT install prefix
    * PATH Default:

The following options control `find_package` paths for CMake-based TPLs:
* HPX_DIR or HPX_ROOT
    * Location of HPX prefix (ROOT) or CMake config file (DIR)
    * PATH Default:

#### Architecture Keywords
Architecture-specific optimizations can be enabled by specifying `-DKokkos_ARCH_X`.

* Kokkos_ARCH_A64FX
    * Whether to optimize for ARMv8.2 with SVE Support
    * BOOL Default: OFF
* Kokkos_ARCH_AMDAVX
    * Whether to optimize for the AMDAVX architecture
    * BOOL Default: OFF
Kokkos_ARCH_AMPERE80
    * Whether to optimize for the NVIDIA Ampere generation CC 8.0
    * BOOL Default: OFF
Kokkos_ARCH_AMPERE86
    * Whether to optimize for the NVIDIA Ampere generation CC 8.6
    * BOOL Default: OFF
* Kokkos_ARCH_ARMV80
    * Whether to optimize for the ARMV80 architecture
    * BOOL Default: OFF
* Kokkos_ARCH_ARMV81
    * Whether to optimize for the ARMV81 architecture
    * BOOL Default: OFF
* Kokkos_ARCH_ARMV8_THUNDERX
    * Whether to optimize for the ARMV8_THUNDERX architecture
    * BOOL Default: OFF
* Kokkos_ARCH_ARMV8_THUNDERX2
    * Whether to optimize for the ARMV8_TX2 architecture
    * BOOL Default: OFF
* Kokkos_ARCH_BDW
    * Whether to optimize for the BDW architecture
    * BOOL Default: OFF
* Kokkos_ARCH_BGQ
    * Whether to optimize for the BGQ architecture
    * BOOL Default: OFF
* Kokkos_ARCH_HSW
    * Whether to optimize for the HSW architecture
    * BOOL Default: OFF
* Kokkos_ARCH_INTEL_GEN
    * Whether to optimize for Intel GPUs Gen9+
    * BOOL Default: OFF
* Kokkos_ARCH_KEPLER30
    * Whether to optimize for the KEPLER30 architecture
    * BOOL Default: OFF
* Kokkos_ARCH_KEPLER32
    * Whether to optimize for the KEPLER32 architecture
    * BOOL Default: OFF
* Kokkos_ARCH_KEPLER35
    * Whether to optimize for the KEPLER35 architecture
    * BOOL Default: OFF
* Kokkos_ARCH_KEPLER37
    * Whether to optimize for the KEPLER37 architecture
    * BOOL Default: OFF
* Kokkos_ARCH_KNC
    * Whether to optimize for the KNC architecture
    * BOOL Default: OFF
* Kokkos_ARCH_KNL
    * Whether to optimize for the KNL architecture
    * BOOL Default: OFF
* Kokkos_ARCH_MAXWELL50
    * Whether to optimize for the MAXWELL50 architecture
    * BOOL Default: OFF
* Kokkos_ARCH_MAXWELL52
    * Whether to optimize for the MAXWELL52 architecture
    * BOOL Default: OFF
* Kokkos_ARCH_MAXWELL53
    * Whether to optimize for the MAXWELL53 architecture
    * BOOL Default: OFF
* Kokkos_ARCH_PASCAL60
    * Whether to optimize for the PASCAL60 architecture
    * BOOL Default: OFF
* Kokkos_ARCH_PASCAL61
    * Whether to optimize for the PASCAL61 architecture
    * BOOL Default: OFF
* Kokkos_ARCH_POWER7
    * Whether to optimize for the POWER7 architecture
    * BOOL Default: OFF
* Kokkos_ARCH_POWER8
    * Whether to optimize for the POWER8 architecture
    * BOOL Default: OFF
* Kokkos_ARCH_POWER9
    * Whether to optimize for the POWER9 architecture
    * BOOL Default: OFF
* Kokkos_ARCH_SKX
    * Whether to optimize for the SKX architecture
    * BOOL Default: OFF
* Kokkos_ARCH_SNB
    * Whether to optimize for the SNB architecture
    * BOOL Default: OFF
* Kokkos_ARCH_TURING75
    * Whether to optimize for the TURING75 architecture
    * BOOL Default: OFF
* Kokkos_ARCH_VOLTA70
    * Whether to optimize for the VOLTA70 architecture
    * BOOL Default: OFF
* Kokkos_ARCH_VOLTA72
    * Whether to optimize for the VOLTA72 architecture
    * BOOL Default: OFF
* Kokkos_ARCH_WSM
    * Whether to optimize for the WSM architecture
    * BOOL Default: OFF
* Kokkos_ARCH_VEGA900
    * Whether to optimize for AMD GPU MI25 GFX900
    * BOOL Default: OFF
* Kokkos_ARCH_VEGA906
    * Whether to optimize for AMD GPU MI50/MI60 GFX906
    * BOOL Default: OFF
* Kokkos_ARCH_VEGA908
    * Whether to optimize for AMD GPU MI100 GFX908
    * BOOL Default: OFF
* Kokkos_ARCH_ZEN
    * Whether to optimize for AMD Zen architecture
    * BOOL Default: OFF
* Kokkos_ARCH_ZEN2
    * Whether to optimize for AMD Zen2 architecture
    * BOOL Default: OFF

## 4.3 Using Trilinos' CMake build system

The Trilinos project (see `trilinos.org` and github for the [source code](https://github.com/trilinos/Trilinos) repository) is an effort to develop algorithms and enabling technologies within an object-oriented software framework for the solution of large-scale, complex multiphysics engineering and scientific problems. Trilinos is organized into packages. Even though Kokkos is a stand-alone software project, Trilinos uses Kokkos extensively. Thus, Trilinos' source code includes Kokkos' source code, and builds Kokkos as part of its build process.

Trilinos' build system uses CMake. Thus, to build Kokkos as part of Trilinos, you must first install CMake (version `3.17` or newer). To enable Kokkos when building Trilinos, set the CMake option `Trilinos_ENABLE_Kokkos`. Trilinos' build system lets packages express dependencies on other packages or external libraries. If you enable any Trilinos package (e.g., Tpetra) that has a required dependency on Kokkos, Trilinos will enable Kokkos automatically. Configuration macros are automatically inferred from Trilinos settings. For example, if the CMake option `Trilinos_ENABLE_OpenMP` is `ON`, Trilinos will define the macro `Kokkos_ENABLE_OPENMP`. Trilinos' build system will autogenerate the previously mentioned `KokkosCore_config.h` file that contains those macros.

Trilinos' CMake build system utilizes Kokkos' build system to set compiler flags, compiler options, architectures, etc. CMake variables `CMAKE_CXX_COMPILER`, `CMAKE_C_COMPILER`, and `CMAKE_FORTRAN_COMPILER` are used to specify the compiler. To configure Trilinos for various architectures, with Kokkos enabled, the CMake variable `Kokkos_ARCH_<ArchCode>` should be set, matching ArchCode to the appropriate architecture as specified in [Architecture Keywords](https://github.com/kokkos/kokkos/wiki/Compiling#architecture-keywords).

For example, `Kokkos_ARCH_HSW` sets the architecture variables for a machine with Intel Haswell CPUs. Also, when setting the `Kokkos_ARCH_<ArchCode>` variable it is not necessary to pass required architecture-specific flags to CMake, for example via the `CMAKE_CXX_FLAGS` variable.

Some Trilinos packages with CUDA support currently require the use of UVM (note UVM is enabled by default when configuring Trilinos with CUDA enabled, unless the user explicitly disables it). To ensure proper compilation and execution for such packages, the environment variables `export CUDA_LAUNCH_BLOCKING=1` and `export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1` must be set.

### Building Trilinos with Kokkos' develop branch

In some cases users may desire to test building Trilinos with Kokkos' develop branch. Note that incompatibilities between Kokkos' develop branch and Trilinos may arise between release cycles and **there is no guarantee of stability for this process between releases**.

To support this setup (without overwriting the Kokkos package in Trilinos), users may

1. Add a symbolic link to the Trilinos source directory pointing to their local Kokkos' repository
2. Include the configure option `Kokkos_SOURCE_DIR_OVERRIDE:STRING=kokkos` in their CMake configuration

The same process above can be applied for KokkosKernels as well, by adding a symbolic link to the local KokkosKernels repository and including the `KokkosKernels_SOURCE_DIR_OVERRIDE:STRING=kokkos-kernels` configuration option.

For builds with CUDA enabled, the path to the `nvcc_wrapper` script should also be specified (as an environment variable for example, i.e. `export CXX=<path-to-kokkos>/bin/nvcc_wrapper` in a non-MPI build)

We refer readers to Trilinos' documentation for further details.

## 4.4 Using Kokkos' GNU Makefile system

The base of the build system is the file `Makefile.kokkos`; it is designed to be included by application Makefiles. It contains logic to (re)generate the `KokkosCore_config.h` file if necessary, build the Kokkos library, and provide updated compiler and linker flags. 

The system can digest several variables that are used to configure Kokkos settings and then parses the variables for Keywords. This allows for multiple options to be given for each variable. The separator doesn't matter as long as it doesn't interact with the Make system. A list of variables, their meaning and options are given in Table 4.4.

A word of caution on where to include the `Makefile.kokkos`: since the embedded Makefiles define targets, it is usually better to include it after the first application target has been defined. Since that target can't use the flags from the embedded Makefiles, it should be a meta target:

    CXX=g++
    default: main
    include Makefile.kokkos
    main: $(KOKKOS_LINK_DEPENDS) $(KOKKOS_CPP_DEPENDS) main.cpp
          $(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) \
          $(KOKKOS_LDFLAGS) $(KOKKOS_LIBS) main.cpp -o main

More example application Makefiles can be found in the tutorial examples under `kokkos/example/tutorial`.

Kokkos provides a script `generate_makefile.bash` which can generate a Makefile for building and installing the library as well as building and running the tests. Please run `generate_makefile.bash --help` to see options. Note that paths given to the script must be absolute paths, and the script must be run with the `bash` shell (the script will do this if it is run directly, i.e., as `./generate_makefile.bash`).

<h4>Table 4.4: Variables for the GNU Makefile</h4>

Variable  | Description
 ---: |:---
`KOKKOS_PATH (IN)` | Path to the Kokkos root or install directory. One can either build against an existing install of Kokkos or use its source directly for an embedded build. In the former case the "Input variables" are set inside the embedded Makefile.kokkos and it is not valid to set them differently in the including Makefile. 
`CUDA_PATH (IN)` | Path to the Cuda toolkit root directory. 
`KOKKOS_DEVICES (IN)` | Execution and Memory Spaces that should be enabled.
`Options`<br>`    Default` | OpenMP, Serial, Pthreads, Cuda <br> OpenMP 
`KOKKOS_ARCH (IN)` | The backend architecture to build for.
`Options` <br><br><br> `Default` | KNL, KNC, SNB, HSW, BDW, Kepler, Kepler30, Kepler35, Kepler37, Maxwell, Maxwell50, Pascal60, Pascal61, ARMv8, ARMv81, ARMv8-ThunderX, BGQ, Power7, Power8 <br><br> (no particular architecture flags are set).
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
`KOKKOS_CPPFLAGS (OUT)` | Preprocessor flags (include directories and defines). Add this to the applications compiler and preprocessor flags.
`KOKKOS_CXXFLAGS (OUT)` | Compiler flags. Add this to the applications compiler flags.
`KOKKOS_LDFLAGS (OUT)` | Linker flags. Add this to the applications linker flags.
`KOKKOS LIBS (OUT)` | Libraries required by Kokkos. Add this to the link line after the linker flags.
`KOKKOS_CPP_DEPENDS (OUT)` |  Dependencies for compilation units which include any Kokkos header files. <br> Add this as a dependency to compilation targets including any Kokkos code.
`KOKKOS_LINK_DEPENDS (OUT)` | Dependencies of an application linking in the Kokkos library. Add this to the dependency list of link targets.
`CXXFLAGS (IN)` | User provided compiler flags which will be used to compile the Kokkos library.
`CXX (IN)` | The compiler used to compile the Kokkos library.

## 4.5 Building for CUDA

Any Kokkos application compiled for CUDA embeds CUDA code via template metaprogramming. Thus, the whole application must be built with a CUDA-capable compiler. (At the moment, the only such compilers are NVIDIA's NVCC and Clang 4.0+) More precisely, every compilation unit containing a Kokkos kernel or a function called from a Kokkos kernel has to be compiled with a CUDA-capable compiler. This includes files containing `Kokkos::View` allocations which call an initialization kernel.

All current versions of the NVCC compiler have shortcomings when used as the main compiler for a project, in particular when part of a complex build system. For example, it does not understand most GCC command-line options, which must be prepended by the `-Xcompiler` flag when calling NVCC. Kokkos comes with a shell script, called `nvcc_wrapper`, that wraps NVCC to address these issues. We intend this as a drop-in replacement for a normal GCC-compatible compiler (e.g., GCC or Intel) in your build system. It analyzes the provided command-line options and prepends them correctly. It also adds the correct flags for compiling generic C++ files containing CUDA code (e.g., `*.cpp, *.cxx,` or `*.CC`). By default `nvcc_wrapper` calls `g++` as the host compiler. You may override this by providing NVCC's `-ccbin` option as a compiler flag. The default can be set by editing the script itself or by setting the environment variable `NVCC_WRAPPER_DEFAULT_COMPILER`.

Many people use a system like Environment Modules (see `http://modules.sourceforge.net/`) to manage their shell environment. When using a module system, it can be useful to provide different versions for different back-end compiler types (e.g., `icpc, pgc++, g++,` and `clang`). To use the `nvcc_wrapper` in conjunction with MPI wrappers, simply overwrite which C++ compiler is called by the MPI wrapper. For example, you can reset OpenMPI's C++ compiler by setting the `OMPI_CXX` environment variable. Make sure that `nvcc_wrapper` calls the host compiler with which the MPI library was compiled.

## 4.6 Execution Space Restrictions

Currently, Kokkos organizes its execution spaces into three categories:

 - Host Serial: A top-level `Serial` execution space with no parallelism or concurrency
 - Host Parallel: Typically a threading model for CPUs, currently: `OpenMP`, `Threads`, and `QThreads`.
 - Device Parallel: Typically an attached GPU, currently: `CUDA`, `OpenMPTarget`, and `ROCm`.

The current Kokkos policy is to allow users, at compile time, to enable *at most one* execution space from each category. This prevents incompatibilities between different spaces in the same category from degrading the user's correctness and performance.


**[[Chapter 5: Initialization|Initialization]]**
