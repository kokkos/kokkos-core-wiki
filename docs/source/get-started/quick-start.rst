Quick Start
===========

This guide is intended to jump start new Kokkos users (and beginners, in particular).


Prerequisites
~~~~~~~~~~~~~

To complete this tutorial, you'll need:

* a compatible operating system (e.g. Linux, macOS, Windows).
* a compatible C++ compiler that supports at least C++17.
* `CMake <https://cmake.org/>`_ and a compatible build tool for building the
  project.

  * Compatible build tools include `Make
    <https://www.gnu.org/software/make/>`_, `Ninja <https://ninja-build.org>`_,
    and others - see `CMake Generators
    <https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html>`_ for
    more information.

See :doc:`requirements` for more information about platforms compatible with
Kokkos and a list of supported versions of compilers and software development
kits (SDKs).

If you don’t already have CMake installed, see the `CMake installation guide
<https://cmake.org/install>`_.


Set up a project
~~~~~~~~~~~~~~~~

CMake uses a file named ``CMakeLists.txt`` to configure the build system for a
project. You’ll use this file to set up your project and declare a dependency
on Kokkos.

First, create a directory for your project:

.. code-block:: sh

  > mkdir MyProject && cd MyProject

Next, you’ll create the ``CMakeLists.txt`` file and declare a dependency on
Kokkos. There are many ways to express dependencies in the CMake ecosystem; in
this tutorial, you’ll use the `FetchContent CMake module
<https://cmake.org/cmake/help/latest/module/FetchContent.html>`_. To do this,
in your project directory (``MyProject``), create a file named
``CMakeLists.txt`` with the following contents:

.. code-block:: cmake

  cmake_minimum_required(VERSION 3.16)
  project(MyProject)
  
  include(FetchContent)
  FetchContent_Declare(
    Kokkos
    URL https://github.com/kokkos/kokkos/archive/refs/tags/4.5.01.zip
  )
  FetchContent_MakeAvailable(Kokkos)

The above configuration declares a dependency on Kokkos which is downloaded
from GitHub.
``4.5.01`` is the Kokkos version to use; we generally recommend using the
`latest available <https://github.com/kokkos/kokkos/releases/latest>`_.

For more information about how to create ``CMakeLists.txt files``, see the
`CMake Tutorial
<https://cmake.org/cmake/help/latest/guide/tutorial/index.html>`_.


Create and run an executable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With Kokkos declared as a dependency, you can use Kokkos code within your own
project.

As an example, create a ``HelloKokkos.cpp`` with the following content:

.. code-block:: c++

  #include <Kokkos_Core.hpp>

  int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
      // Allocate a 1-dimensional view of integers
      Kokkos::View<int*> v("v", 5);
      // Fill view with sequentially increasing values v=[0,1,2,3,4]
      Kokkos::parallel_for("fill", 5, KOKKOS_LAMBDA(int i) { v(i) = i; });
      // Compute accumulated sum of v's elements r=0+1+2+3+4
      int r;
      Kokkos::parallel_reduce(
        "accumulate", 5,
        KOKKOS_LAMBDA(int i, int& partial_r) { partial_r += v(i); }, r);
      // Check the result
      KOKKOS_ASSERT(r == 10);
    }
    Kokkos::printf("Goodbye World\n");
    Kokkos::finalize();
    return 0;
  }

The above program code includes the main Kokkos header file and demonstrates
how to initialize and finalize the Kokkos execution environment.

To build the code, add the following couple lines to the end of your
``CMakeLists.txt`` file:

.. code-block:: cmake

  add_executable(HelloKokkos HelloKokkos.cpp)
  target_link_libraries(HelloKokkos Kokkos::kokkos)


The above configuration declares the executable you want to build
(``HelloKokkos``), and links it to Kokkos.

Now you can build and run your Kokkos program.

Start with calling ``cmake`` to configure the project and generate a native
build system:

.. code-block:: sh

  MyProject> cmake -B builddir
  -- The C compiler identification is GNU 10.2.1
  -- The CXX compiler identification is GNU 10.2.1
  ...
  -- Build files have been written to: .../MyProject/builddir


.. note::

  If you want to target a NVIDIA GPU, you will need to pass an extra
  ``-DKokkos_ENABLE_CUDA=ON`` argument to the cmake command above. For an AMD
  or an Intel GPU, you would use ``-DKokkos_ENABLE_HIP=ON`` or
  ``-DKokkos_ENABLE_SYCL=ON`` respectively. For a list of options and variables
  available at configuration time, see :doc:`configuration-guide`.


Then invoke that build system to actually compile/link the project:

.. code-block:: sh

  MyProject> cmake --build builddir
  Scanning dependencies of target ...
  ...
  [100%] Built target HelloKokkos

Finally try to use the newly built ``HelloKokkos``:

.. code-block:: sh

  MyProject> cd builddir

  MyProject/builddir> HelloKokkos
  Goodbye World

.. note::

   Depending on your shell, the correct syntax may be ``HelloKokkos``,
   ``./HelloKokkos`` or ``.\HelloKokkos``.

Congratulations! You’ve successfully built and run a test binary using Kokkos.


Getting help
~~~~~~~~~~~~

If you need additional help getting started, please join the `Kokkos Slack
Workspace <https://kokkosteam.slack.com>`_. If you have trouble signing up see the
:ref:`FAQ entry on how to join <join-slack-workspace>`.
