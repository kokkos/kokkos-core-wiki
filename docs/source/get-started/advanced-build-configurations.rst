Advanced Build Configurations
====================================

This document describes which advanced methods are available to integrate
Kokkos into a project.

Kokkos provides the ``Kokkos::kokkos`` target, which simplifies the
process by automatically handling necessary include directories, link
libraries, compiler options, and other usage requirements.

Here are several integration methods, each with its own advantages:

1. Kokkos traverses all targets unless instructed otherwise
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In general, Kokkos reserves the right to traverse all targets that have been defined in your project,
check them for `PRIVATE` and `PUBLIC` dependency to Kokkos and set properties like
the compile language, architecture, compiler, etc. on your sources and targets.
This is required as Kokkos and CMake need to set compiler and linker settings so your source files 
can be compiled for the backend and architecture you specified when building Kokkos itself.

To do this, `find_package(Kokkos)` injects a call that CMake executes when it reaches
the end of the top-level `CMakeLists.txt` file.
This call only changes properties of source files or targets if the target links to Kokkos
and the source file has a file ending associated with c++.
The project in `examples/build_installed/complex` shows this strategy for a complex project
with multiple, nested libraries.

2. Excluding directories from being traversed by Kokkos
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your project uses multiple libraries that depend on Kokkos or 
not all your source files can be compiled with the compiler Kokkos requires,
you might require fine control on the rules used for building your project.
To opt-out of Kokkos setting the properties on sources and targets, we export the CMake function
`kokkos_exclude_from_setting_build_properties` which takes `GLOBAL` as an option or a list of directories with the keyword `DIRECTORY`.
To opt-in individual sources and targets, Kokkos defines the CMake function `kokkos_set_source_and_target_properties`,
which sets the properties according with what the found Kokkos package requires.
The function accepts the option `GLOBAL` and the keywords `DIRECTORY`, `SOURCE`, `TARGET`, `LINK_ONLY_TARGET` which take lists.
These two options together allow fine grained control to the level of individual source files and targets.
An example of setting the properties at an individual level can be found in `examples/build_installed/complex_mark_kokkos_files`.

3. Guard your own library against downstream users excluding directories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Setting these properties at indiviual source and target level can be tedious.
Consider the situation of a library that uses Kokkos in its interfaces and that is used downstream,
possibly together with other libraries that might link to Kokkos.
This can create a situation where downstream users opt-out of Kokkos setting source and target properties,
but because your library exposes Kokkos in its interfaces, the sources and targets using your library depend on
the correct compiler settings to work.
To adress this, Kokkos provides the CMake function `kokkos_defer_set_dependent_library_properties` which takes a `LIBRARY` keyword.
This function injects a call that CMake executes when it reaces the end of the top-level `CMakeLists.txt`.
In the injected call, CMake traverses all targets defined in the project and sets the compile and link properties depending on if your library depends `PUBLIC` or `PRIVATE` on Kokkos.
This propagates the source and target properites Kokkos requires to the dependants of your library
and is independent of the directories excluded by `kokkos_exclude_from_setting_build_properties`.
An example can be found in `examples/build_installed/complex_mark_library`.   
