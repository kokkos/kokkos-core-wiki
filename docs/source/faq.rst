FAQ
###

.. _join-slack-workspace:

How do I join the Kokkos slack channel?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You can find the slack channel at `kokkosteam.slack.com <https://kokkosteam.slack.com>`_. Register a new account with your email. We reached the limit of whitelisted organizations, but every member of the Kokkos Slack workspace can invite more people. If no one you know is in the Slack workspace you can contact the Kokkos maintainers (their emails are in the LICENSE file).

How do I compile Kokkos with C++20 or C++23?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When configuring Kokkos with cmake, add the flag ``-DCMAKE_CXX_STANDARD=20`` (or ``23``). Ensure that the flag is also set for any downstream applications.

.. _setup-debug-build:

How do I set up a debug build?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Debug behavior in Kokoks is both affected by the build type and additional debug CMake options:

1. CMake build type and compiler flags:

   * ``CMAKE_BUILD_TYPE=Debug``: Commonly enables flags for debug symbols (``-g``) without specifying optimization flags. Enables ``Kokkos_ENABLE_DEBUG`` by default. Enables ``KOKKOS_ASSERT``.

   * ``CMAKE_BUILD_TYPE=RelWithDebInfo``: Commonly enables flags for debug symbols (``-g``) with optimization flags (``-O2``).

   **NVCC Specifics:** To get full device debug symbols, you must manually add ``-DCMAKE_CXX_FLAGS="-G"`` and use ``nvcc_wrapper`` as ``CMAKE_CXX_COMPILER``. If building with ``-DKokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE=ON``, specify ``-DCMAKE_CUDA_FLAGS="-G"`` instead.

   .. warning:: ``-G`` disables nearly all GPU optimizations and will significantly slow down your kernels.

2. CMake options for tweaking debug settings:

   ``Kokkos_ENABLE_DEBUG``

   * Enables ``KOKKOS_ASSERT``. By default activated when compiling with ``CMAKE_BUILD_TYPE=Debug``.

   * **NVCC Specifics:** Only ``-lineinfo`` is added for device debug symbols. This provides source-line information for profilers (like Nsight) without the massive performance penalty of ``-G``.

   ``Kokkos_ENABLE_DEBUG_BOUNDS_CHECK``

   * Enables out-of-bounds checks in Views. Due to high overhead, this option is disabled for all build types by default.

   * Enabling this option implies synchronization after every kernel with the CUDA or HIP backend.
