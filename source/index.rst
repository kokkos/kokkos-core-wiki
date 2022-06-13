.. role:: raw-html-m2r(raw)
   :format: html

.. include:: mydefs.rst

Kokkos: The Programming Model 
=============================

.. admonition:: :medium:`C++ Performance Portability Programming Model`
    :class: important

    Kokkos Core implements a programming model in C++ for writing performance portable applications targeting all major HPC platforms. For that purpose it provides abstractions for both parallel execution of code and data management. Kokkos is designed to target complex node architectures with N-level memory hierarchies and multiple types of execution resources. It currently can use CUDA, HIP, SYCL, HPX, OpenMP and C++ threads as backend programming models with several other backends development.


The `Kokkos EcoSystem <https://github.com/kokkos>`_ includes: 

.. list-table::
   :widths: 30 50 20
   :header-rows: 1
   :align: left

   * - Name
     - Info
     - 

   * - :packnameindexpage:`kokkos`
     - this library
     - `Github link <https://github.com/kokkos/kokkos>`__

   * - :packnameindexpage:`kokkos-kernels`
     - Sparse, dense, batched math kernels
     - `Github link <https://github.com/kokkos/kokkos-kernels>`__

   * - :packnameindexpage:`kokkos-tools`
     - Profiling and debugging tools
     - `Github link <https://github.com/kokkos/kokkos-tools>`__

   * - :packnameindexpage:`pykokkos`
     - Provides Python bindings to the Kokkos performance portable parallel programming.
     - `Github link <https://github.com/kokkos/pykokkos>`__


Questions?
----------

Find us on Slack: https://kokkosteam.slack.com or
open an issue on `github <https://github.com/kokkos/kokkos/issues>`_.

|
|
|
|


.. toctree::
   :maxdepth: 1

   programmingguide
   requirements
   building
   keywords
   ./API/core-index
   ./API/containers-index
   ./API/algorithms-index
   ./API/std-algorithms-index
   ./API/alphabetical
   usecases
   testing-and-issue-tracking
   Tutorials <https://github.com/kokkos/kokkos-tutorials>
   videolectures
   GitHub Repo <https://github.com/kokkos/kokkos>
   Open an issue/feature req. <https://github.com/kokkos/kokkos/issues>
   citation
   license
