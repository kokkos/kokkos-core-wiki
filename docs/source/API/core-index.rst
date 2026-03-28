Core API
########

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Reducer
     - Description
   * - :doc:`Initialization and Finalization <core/Initialize-and-Finalize>`
     - Initialization and finalization of Kokkos.
   * - :doc:`View and related <core/View>`
     - Kokkos MultiDimensional View class and related free functions.
   * - :doc:`Parallel Execution/Dispatch <core/ParallelDispatch>`
     - Parallel Execution Dispatch.
   * - :doc:`Built-in Reducers <core/builtin_reducers>`
     - Built-in Reducers
   * - :doc:`Execution Policies <core/Execution-Policies>`
     - Execution policies.
   * - :doc:`Spaces <core/Spaces>`
     - Description of Memory and Execution Spaces.
   * - :doc:`Task-Parallelism <core/Task-Parallelism>`
     - Creating and dispatching Task Graphs.
   * - :doc:`MultiGPU Support <core/MultiGPUSupport>`
     - Launching kernels on multiple GPUs from one process.
   * - :doc:`Atomics <core/atomics>`
     - Atomics
   * - :doc:`Numerics <core/Numerics>`
     - Common mathematical functions, mathematical constants, numeric traits,
       complex numbers, half-precision floating-point types.
   * - :doc:`C-style memory management <core/c_style_memory_management>`
     - C-style memory management
   * - :doc:`Traits <core/Traits>`
     - Traits
   * - :doc:`Kokkos Concepts <core/KokkosConcepts>`
     - Kokkos Concepts
   * - :doc:`STL Compatibility Issues <core/STL-Compatibility>`
     - Ports of standard C++ capabilities, which otherwise do not work on various hardware platforms.
   * - :doc:`Utilities <core/Utilities>`
     - Utility functionality part of Kokkos Core.
   * - :doc:`Detection Idiom <core/Detection-Idiom>`
     - Used to recognize, in an SFINAE-friendly way, the validity of any C++ expression.
   * - :doc:`Macros <core/Macros>`
     - Global macros defined by Kokkos, used for architectures, general settings, etc.

.. toctree::
   :hidden:
   :maxdepth: 1

   ./core/Initialize-and-Finalize
   ./core/View
   ./core/ParallelDispatch
   ./core/builtin_reducers
   ./core/Execution-Policies
   ./core/Spaces
   ./core/Task-Parallelism
   ./core/MultiGPUSupport
   ./core/atomics
   ./core/Numerics
   ./core/c_style_memory_management
   ./core/Traits
   ./core/KokkosConcepts
   ./core/STL-Compatibility
   ./core/Utilities
   ./core/Detection-Idiom
   ./core/Macros
   ./core/Profiling
