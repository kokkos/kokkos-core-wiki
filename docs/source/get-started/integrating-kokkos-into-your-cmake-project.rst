Integrating Kokkos into Your Project
====================================

This document describes how to integrate the Kokkos library into your CMake
project.

Kokkos provides the ``Kokkos::kokkos`` target, which simplifies the
process by automatically handling necessary include directories, link
libraries, compiler options, and other usage requirements.

Here are several integration methods, each with its own advantages:

1. External Kokkos (Recommended for most users)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The recommended approach is to use Kokkos as an external dependency. This
allows for easier management and updates.  Use CMake's
`find_package() <https://cmake.org/cmake/help/latest/command/find_package.html>`_
command to locate and link against an existing Kokkos installation:

.. code-block:: cmake

  find_package(Kokkos 4.2 REQUIRED CONFIG) # Find Kokkos version 4.2 or any newer version
  # ...
  target_link_libraries(MyTarget PRIVATE Kokkos::kokkos)

* ``find_package(Kokkos ...)`` searches for a ``KokkosConfig.cmake`` file, which is
  generated when Kokkos is built and installed. This file contains the
  necessary information for linking against Kokkos.
* The ``4.2`` argument specifies the minimum required Kokkos version. It's
  optional but recommended for ensuring compatibility.  Note that it will setup
  your project accept any newer version, including the next major release
  (e.g., ``5.0``).

  * If you need to strictly stay on Kokkos ``4.x`` (to avoid breaking changes
    in ``5.0``), you must specify a version range instead:

    .. code-block:: cmake

      # Find Kokkos 4.2+, but stop before 5.0
      find_package(Kokkos 4.2...<5.0 REQUIRED CONFIG)

* ``Kokkos::kokkos`` is the namespaced imported target that provides all
  necessary build flags.
* The ``CONFIG`` keyword tells CMake to use the configuration files.

You can install Kokkos separately and then point CMake to its location using
the ``Kokkos_ROOT`` variable.

.. code-block:: sh

  MyProject> cmake -DKokkos_ROOT=/path/to/kokkos/install/dir -B builddir


2. Embedded Kokkos via ``add_subdirectory()`` and Git Submodules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This method embeds the Kokkos source code directly into your project.  It's
useful when you need very tight control over the Kokkos version or when you
can't install Kokkos separately.

1. Add Kokkos as a `Git submodule <https://git-scm.com/book/en/v2/Git-Tools-Submodules>`_:

.. code-block:: sh

  MyProject> git submodule add -b 4.5.01 https://github.com/kokkos/kokkos.git tpls/kokkos
  MyProject> git commit -m 'Adding Kokkos v4.5.1 as a submodule'


``tpls/kokkos/`` should now contain the full Kokkos source tree.

2. Use ``add_subdirectory()`` in your CMakeLists.txt:

.. code-block:: cmake

  add_subdirectory(tpls/kokkos)
  # ...
  target_link_libraries(MyTarget PRIVATE Kokkos::kokkos)


3. Embedded Kokkos via ``FetchContent``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   
`FetchContent <https://cmake.org/cmake/help/latest/module/FetchContent.html>`_
simplifies the process of downloading and including Kokkos as a dependency
during the CMake configuration stage.

.. code-block:: cmake

  include(FetchContent)
  FetchContent_Declare(
      Kokkos
      URL      https://github.com/kokkos/kokkos/releases/download/4.5.01/kokkos-4.5.01.tar.gz
      URL_HASH SHA256=52d003ffbbe05f30c89966e4009c017efb1662b02b2b73190670d3418719564c
  )
  FetchContent_MakeAvailable(Kokkos)
  # ...
  target_link_libraries(MyTarget PRIVATE Kokkos::kokkos)


* ``URL_HASH`` is highly recommended for verifying the integrity of the
  downloaded archive. You can find the SHA256 checksums for Kokkos releases in
  the ``kokkos-X.Y.Z-SHA-256.txt`` file on the `Kokkos releases page
  <https://github.com/kokkos/kokkos/releases>`_.


4. Supporting Both External and Embedded Kokkos
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   

This approach allows your project to use either an external Kokkos installation
or an embedded version, providing flexibility for different build environments.

.. code-block:: cmake

  find_package(Kokkos CONFIG) # Try to find Kokkos externally
  if(Kokkos_FOUND)
      message(STATUS "Found Kokkos: ${Kokkos_DIR} (version \"${Kokkos_VERSION}\")")
  else()
      message(STATUS "Kokkos not found externally. Fetching via FetchContent.")
      include(FetchContent)
      FetchContent_Declare(
          Kokkos
          URL https://github.com/kokkos/kokkos/archive/refs/tags/4.4.01.tar.gz
      )
      FetchContent_MakeAvailable(Kokkos)
  endif()
  # ...
  target_link_libraries(MyTarget PRIVATE Kokkos::kokkos)


Controlling the Kokkos integration:

* `CMAKE_DISABLE_FIND_PACKAGE_Kokkos <https://cmake.org/cmake/help/latest/variable/CMAKE_DISABLE_FIND_PACKAGE_PackageName.html>`_:
  Set this variable to ``TRUE`` to force the use of the embedded Kokkos, even if
  an external installation is found.
* `CMAKE_REQUIRE_FIND_PACKAGE_Kokkos <https://cmake.org/cmake/help/latest/variable/CMAKE_REQUIRE_FIND_PACKAGE_PackageName.html>`_:
  Set this variable to ``TRUE`` to require an external Kokkos installation. The
  build will fail if Kokkos is not found.
* ``Kokkos_ROOT``: Use this variable to specify the directory where CMake should
  search for Kokkos when using ``find_package()``.

For example:

.. code-block:: sh

  cmake -DCMAKE_REQUIRE_FIND_PACKAGE_Kokkos=ON -DKokkos_ROOT=/path/to/kokkos/install/dir

or

.. code-block:: sh

  cmake -DCMAKE_DISABLE_FIND_PACKAGE_Kokkos=ON
