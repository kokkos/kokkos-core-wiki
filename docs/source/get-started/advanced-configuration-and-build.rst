Advanced Configuration and Build
================================

nvcc_wrapper: Why do we have it and what does it do
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In general (except when being configured with ``Kokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE=ON``), performance-portable code depending on Kokkos identifies itself as ``CXX`` code in ``CMake``.
This implies that the compiler used by ``CMake`` to compile ``CXX`` code must be able to compile the code for the backend and architecture enabled in Kokkos.

For the ``CUDA`` backend this implies setting ``CMAKE_CXX_COMPILER=nvcc`` during configuration. But ``nvcc`` needed separate flags for the host and the device compilation (newer ``nvcc`` versions have improved support for unknown flags).
This requirement of ``nvcc`` for separate flags implies that other libraries that are linked to the same target also need to adhere to do this.

To help users with this problem, Kokkos comes with a small ``bash`` script called ``nvcc_wrapper`` located in the ``bin`` subdirectory (At the current state there is no way to use this script in ``MSVC`` builds. Please see the section on compiling Kokkos in CMake language mode below).
This script has two functions. It redirects compile and link commands to ``nvcc``, and it sorts the given compiler and linker flags into the ones for the host and the device compiler.

To use it, set ``CMAKE_CXX_COMPILER=<path_to_kokkos>/bin/nvcc_wrapper``, replacing ``<path_to_kokkos>`` with the path of the Kokkos you are using. If you do not do that, Kokkos will use ``kokkos_launch_compiler`` (see next section) to redirect calls to ``nvcc_wrapper`` automatically.

kokkos_launch_compiler: What does it do and how to control it
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In complex software projects that rely on multiple libraries, it can occur that the compiler Kokkos requires for the enabled backend can not compile one of the other libraries the project depends on.
To work around these situations, Kokkos introduced another ``bash`` script called ``kokkos_launch_compiler``.
This script **only** redirects compiler and linker commands that compile a ``C++`` file that uses Kokkos to a compiler that can compile Kokkos code (e.g. ``nvcc_wrapper`` for the ``CUDA`` backend). Compiler and linker commands of ``C++`` files that don't use Kokkos, or files in different languages will not be redirected.

This script, located in the ``bin`` subdirectory, is meant to be used like a compiler launcher in ``CMake``.
But (except when being configured with ``Kokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE=ON``) Kokkos will try to detect if the ``CXX`` compiler that ``CMake`` uses can compile the code for the enabled backend. If the ``CXX`` compiler can **not** compile the backend code, Kokkos automatically uses ``kokkos_launch_compiler``. The idea is to help users create performance-portable libraries that seamlessly integrate into complex software projects.

Although, this covers most usecases, Kokkos provides ways for users to request ``kokkos_launch_compiler`` to be used **always** or **never**.
To always use ``kokkos_launch_compiler``, users can ask for the ``launch_compiler`` component when calling ``find_packlage``:

.. code-block:: sh

   find_package(Kokkos REQUIRED COMPONENTS launch_compiler)

In contrast, when users want that ``kokkos_launch_compiler`` to never be used:

.. code-block:: sh

   find_package(Kokkos REQUIRED COMPONENTS separable_compilation)

Compile in the CMake language of the backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the backends ``CUDA`` and ``HIP``, that have dedicated ``CMake`` languages corresponding to them, Kokkos can be configured to only set compiler and linker flags in the respective ``CMake`` language. This effectively causes Kokkos to identify as a ``CUDA`` or ``HIP`` library to ``CMake`` (**instead** of identifying as a ``CXX`` library). This does allow additional host backends to be enabled (e.g. ``CUDA`` and ``OPENMP`` and ``Serial`` can be compiled in ``CUDA`` language mode).
This implies, that files that link to Kokkos also have to identify as the respective ``CMake`` language, and need to be compiled for the architecture that is enabled in Kokkos.
To enable this behavior, enable the option `Kokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE <configuration-guide.html#backend-specific-options>`_. For a detailed example and explanation check the notes on the option.

Compile in multiple languages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Complex software projects might use multiple libraries that use Kokkos. If one of these uses Kokkos in the ``CMake`` language of the backend (e.g. ``Cuda`` or ``HIP``), while other libraries use Kokkos as a ``CXX`` library, they need a Kokkos that can be compiled in both modes.
To allow this, Kokkos provides the option `Kokkos_ENABLE_MULTIPLE_CMAKE_LANGUAGES <configuration-guide.html#backend-specific-options>`_. For details and requirements check the notes on the option.
