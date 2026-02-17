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
To set up a debug build in Kokkos, you can choose between a full symbolic debug (slowest) or a "fast debug" that keeps optimizations while enabling safety checks.

1. **Standard (Full) Debug Build**

   Use this for full source-level debugging (e.g., using GDB, LLDB, or cuda-gdb).

   * **CMake Flags:** ``-DCMAKE_BUILD_TYPE=Debug``

   * **Effects**: Sets ``-O0`` (no optimization), adds host debug symbols (``-g``), and enables ``KOKKOS_ASSERT``.

   * **CUDA Specifics:** To get full device-side symbolic information, you must manually add ``-DCMAKE_CXX_FLAGS="-G"`` (unless using Clang) and use ``nvcc_wrapper`` as ``CMAKE_CXX_COMPILER``.

   .. warning:: ``-G`` disables nearly all GPU optimizations and will significantly slow down your kernels.


2. **Fast Debug**

   This is a good compromise for development. It keeps optimizations enabled but turns on internal safety logic.

   * **CMake Flags:** ``-DCMAKE_BUILD_TYPE=RelWithDebInfo`` and ``-DKokkos_ENABLE_DEBUG=ON``

   * **Effects:** Keeps code fast (``-O2``) and enables ``KOKKOS_ASSERT``.

   * **CUDA Specifics:** Setting ``-DKokkos_ENABLE_DEBUG=ON`` automatically adds ``-lineinfo`` with NVCC. This provides source-line information for profilers (like Nsight) without the massive performance penalty of ``-G``. No extra flags are required for "fast" GPU debugging.


3. **Bounds Checking**

   By default, neither of the above options enables bounds checking due to the high overhead. To catch "out-of-bounds" errors in your Views:

   * **CMake Flag:** ``-DKokkos_ENABLE_DEBUG_BOUNDS_CHECK=ON``
