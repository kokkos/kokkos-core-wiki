# Build, Install and Use

## Kokkos Philosophy
Kokkos provides a modern CMake style build system.
As C++ continues to develop for C++20 and beyond, CMake is likely to provide the most robust support
for C++.  Applications heavily leveraging Kokkos are strongly encouraged to use a CMake build system.

You can either use Kokkos as an installed package (encouraged) or use Kokkos in-tree in your project.
Modern CMake is exceedingly simple at a high-level (with the devil in the details).
Once Kokkos is installed In your `CMakeLists.txt` simply use:
````cmake
find_package(Kokkos REQUIRED)
````
Then for every executable or library in your project:
````cmake
target_link_libraries(myTarget Kokkos::kokkos)
````
That's it! There is no checking Kokkos preprocessor, compiler, or linker flags.
Kokkos propagates all the necessary flags to your project.
This means not only is linking to Kokkos easy, but Kokkos itself can actually configure compiler and linker flags for *your*
project.
When configuring your project just set:
````bash
> cmake ${srcdir} \
  -DKokkos_ROOT=${kokkos_install_prefix} \
  -DCMAKE_CXX_COMPILER=${compiler_used_to_build_kokkos}
````
Note: You may need the following if using some versions of CMake (e.g. 3.12):
````cmake
cmake_policy(SET CMP0074 NEW)
````
If building in-tree, there is no `find_package`. You can use `add_subdirectory(kokkos)` with the Kokkos source and again just link with `target_link_libraries(Kokkos::kokkos)`.
The examples in `examples/build_installed` and `examples/build_in_tree` can help get you started.

## Configuring CMake to buid Kokkos
A very basic installation of Kokkos is done with:
````bash
> cmake ${srcdir} \
 -DCMAKE_CXX_COMPILER=g++ \
 -DCMAKE_INSTALL_PREFIX=${kokkos_install_folder}
````
which builds and installed a default Kokkos when you run `make install`.
There are numerous device backends, options, and architecture-specific optimizations that can be configured, e.g.
````bash
> cmake ${srcdir} \
 -DCMAKE_CXX_COMPILER=g++ \
 -DCMAKE_INSTALL_PREFIX=${kokkos_install_folder} \
 -DKokkos_ENABLE_OPENMP=ON
````
which activates the OpenMP backend. All the options controlling device backends, options, architectures, and third-party libraries (TPLs) are given in [CMake Keywords](../keywords).

## Advanced ways of integrating Kokkos in a project
(since Kokkos 4.5)
In general, Kokkos reserves the right to traverse all targets that have been defined in your project, check them for `PRIVATE` and `PUBLIC` dependency to Kokkos and set properties like the compile language, architecture, compiler, etc. on your sources and targets.
This is required as Kokkos and CMake need to set compiler and linker settings so your source files can be compiled for the architecture you specified when building Kokkos.
To do this, `find_package(Kokkos)` injects a call that CMake executes when it reaches the end of the top-level `CMakeLists.txt` file. This call only changes properties if the target links to Kokkos and only for `.cpp` and `.cxx` source files.
`examples/build_installed/complex` shows this strategy for a complex project.

If your project uses multiple libraries that depend on Kokkos, not all your source files can be compiled with the compiler Kokkos requires, or you write a library that exposes kokkos in interfaces, you might require fine control on the rules used for building your project.
To opt-out of Kokkos setting the properties on sources and targets, we export the CMake function `kokkos_exclude_from_setting_build_properties` which takes `GLOBAL` as an option or a list of directories with the keyword `DIRECTORY`.
To opt-in individual sources and targets, Kokkos defines the CMake function `kokkos_set_source_and_target_properties` which sets the properties in alignment with the found Kokkos package.
The function accepts the option `GLOBAL` and the keywords `DIRECTORY`, `SOURCE`, `TARGET`, `LINK_ONLY_TARGET` which take lists.
These two options together allow fine grained control on the level of individual source files and targets.
An example of setting the properties at an individual level can be found in `examples/build_installed/complex_mark_kokkos_files`.

Setting these properties at indiviual source and target level can be tedious. Consider the situation of writing a library that uses Kokkos in its interfaces and that is used by users downstream, possibly together with other libraries that might link to Kokkos.
This can create a situation where downstream users opt-out of Kokkos setting source and target properties but because your library exposes Kokkos in its interfaces, the sources and targets using your library depend on the correct compiler settings to work.
To adress this, Kokkos provides the CMake function `kokkos_defer_set_dependent_library_properties` which takes a `LIBRARY` keyword and a `PUBLIC` option. This function injects a call that CMake executes when it reaces the end of the top-level `CMakeLists.txt`.
In the injected call, CMake traverses all targets defined in the project and sets the corresponding Kokkos compile and link properties. The option `PUBLIC` lets you specify if your library links `PUBLIC` to Kokkos, without the option it assumes `PRIVATE`.
This propagates the properites Kokkos requires to the dependants of your library and is independent of the directories excluded by `kokkos_exclude_from_setting_build_properties`.
An example can be found in `examples/build_installed/complex_mark_library`.


## Known Issues<a name="KnownIssues"></a>

### Cray

* The Cray compiler wrappers do static linking by default. This seems to break the Kokkos build. You will likely need to set the environment variable `CRAYPE_LINK_TYPE=dynamic` in order to link correctly. Kokkos warns during configure if this is missing.
* The Cray compiler identifies to CMake as Clang, but it sometimes has its own flags that differ from Clang. We try to include all exceptions, but flag errors may occur in which a Clang-specific flag is passed that the Cray compiler does not recognize.

### Fortran

* In a mixed C++/Fortran code, CMake will use the C++ linker by default. If you override this behavior and use Fortran as the link language, the link may break because Kokkos adds linker flags expecting the linker to be C++. Prior to CMake 3.18, Kokkos has no way of detecting in downstream projects that the linker was changed to Fortran.  From CMake 3.18, Kokkos can use generator expressions to avoid adding flags when the linker is not C++. Note: Kokkos will not add any linker flags in this Fortran case. The user will be entirely on their own to add the appropriate linker flags.

## Raw Makefile

Raw Makefiles are only supported via inline builds. See below.

## Inline Builds vs. Installed Package
For individual projects, it may be preferable to build Kokkos inline rather than link to an installed package.
The main reason is that you may otherwise need many different
configurations of Kokkos installed depending on the required compile time
features an application needs. For example there is only one default
execution space, which means you need different installations to have OpenMP
or C++ threads as the default space. Also for the CUDA backend there are certain
choices, such as allowing relocatable device code, which must be made at
installation time. Building Kokkos inline uses largely the same process
as compiling an application against an installed Kokkos library.

For CMake, this means copying over the Kokkos source code into your project and adding `add_subdirectory(kokkos)` to your CMakeLists.txt.

For raw Makefiles, see the example benchmarks/bytes_and_flops/Makefile which can be used with an installed library and or an inline build.

## Kokkos and CUDA UVM

Kokkos does support UVM as a specific memory space called CudaUVMSpace.
Allocations made with that space are accessible from host and device.
You can tell Kokkos to use that as the default space for Cuda allocations.
In either case UVM comes with a number of restrictions:
* You can't access allocations on the host while a kernel is potentially
running. This will lead to segfaults. To avoid that you either need to
call Kokkos::Cuda::fence() (or just Kokkos::fence()), after kernels, or
you can set the environment variable CUDA_LAUNCH_BLOCKING=1.
* In multi socket multi GPU machines without NVLINK, UVM defaults
to using zero copy allocations for technical reasons related to using multiple
GPUs from the same process. If an executable doesn't do that (e.g. each
MPI rank of an application uses a single GPU [can be the same GPU for
multiple MPI ranks]) you can set CUDA_MANAGED_FORCE_DEVICE_ALLOC=1.
This will enforce proper UVM allocations, but can lead to errors if
more than a single GPU is used by a single process.

## Spack
An alternative to manually building with the CMake is to use the Spack package manager.
Make sure you have downloaded [Spack](https://github.com/spack/spack).
The easiest way to configure the Spack environment is:
````bash
> source spack/share/spack/setup-env.sh
````
with other scripts available for other shells.

You can display information about how to install packages with:
````bash
> spack info kokkos
````
A basic installation would be done as:
````bash
> spack install kokkos
````
Spack allows options and and compilers to be tuned in the install command.
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
<!-- More details can be found in the [Spack README](Spack.md) -->

### Spack Development
Spack currently installs packages to a location determined by a unique hash. This hash name is not really "human readable".
Generally, Spack usage should never really require you to reference the computer-generated unique install folder.
If you must know, you can locate Spack Kokkos installations with:
````bash
> spack find -p kokkos ...
````
where `...` is the unique spec identifying the particular Kokkos configuration and version.

A better way to use Spack for doing Kokkos development is the dev-build feature of Spack.
For dev-build details, try using `spack test run`.

<!-- consult the kokkos-spack repository [README](https://github.com/kokkos/kokkos-spack/blob/master/README.md). -->
