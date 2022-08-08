.. role:: raw-html-m2r(raw)
   :format: html

.. include:: mydefs.rst

Kokkos: The Programming Model
=============================

.. admonition:: :medium:`C++ Performance Portability Programming Model`
    :class: important

    :medium:`Kokkos Core implements a programming model in C++ for writing performance portable applications targeting all major HPC platforms. For that purpose it provides abstractions for both parallel execution of code and data management. Kokkos is designed to target complex node architectures with N-level memory hierarchies and multiple types of execution resources. It currently can use CUDA, HIP, SYCL, HPX, OpenMP and C++ threads as backend programming models with several other backends development.`

The `Kokkos EcoSystem <https://github.com/kokkos>`_ includes:

.. list-table::
   :widths: 30 50 20
   :header-rows: 1
   :align: left

   * - Name
     - Info
     -

   * - ``kokkos``
     - (this library) Programming Model - Parallel Execution and Memory Abstraction
     - `Github link <https://github.com/kokkos/kokkos>`__

   * - ``kokkos-kernels``
     - Sparse, dense, batched math kernels
     - `Github link <https://github.com/kokkos/kokkos-kernels>`__

   * - ``kokkos-tools``
     - Profiling and debugging tools
     - `Github link <https://github.com/kokkos/kokkos-tools>`__

   * - ``pykokkos``
     - Provides Python bindings to the Kokkos performance portable parallel programming.
     - `Github link <https://github.com/kokkos/pykokkos>`__

   * - ``kokkos-remote-spaces``
     - Shared memory semantics across multiple processes
     - `Github link <https://github.com/kokkos/kokkos-remote-spaces>`__

   * - ``kokkos-resilience``
     - Resilience and Checkpointing Extensions for Kokkos
     - `Github link <https://github.com/kokkos/kokkos-resilience>`__

Related Work for the C++ standard library
-----------------------------------------

Relevant and related efforts include:

.. list-table::
   :widths: 20 45 20 15
   :header-rows: 1
   :align: left

   * - Name
     - Info
     - Proposal
     -

   * - ``mdspan``
     - Reference implementation of mdspan targeting C++23
     - `P0009 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p0009r16.html>`__
     - `Github link <https://github.com/kokkos/mdspan>`__

   * - ``stdBLAS``
     - Reference Implementation for stdBLAS
     - `P1673 <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p1673r8.html>`__
     - `Github link <https://github.com/kokkos/stdBLAS>`__

Questions?
----------

Find us on Slack: https://kokkosteam.slack.com or
open an issue on `github <https://github.com/kokkos/kokkos/issues>`_.

Website Content
---------------

.. toctree::
   :maxdepth: 1

   programmingguide
   requirements
   building
   keywords
   ./API/core-index
   ./API/containers-index
   ./API/algorithms-index
   ./API/alphabetical
   usecases
   testing-and-issue-tracking
   Tutorials <https://github.com/kokkos/kokkos-tutorials>
   videolectures
   GitHub Repo <https://github.com/kokkos/kokkos>
   contributing
   citation
   license
