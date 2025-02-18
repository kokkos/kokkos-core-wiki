Core API
########

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Reducer
     - Description
   * - `Initialization and Finalization <core/Initialize-and-Finalize.html>`__
     - Initialization and finalization of Kokkos.
   * - `View and related <core/View.html>`__
     - Kokkos MultiDimensional View class and related free functions.
   * - `Parallel Execution/Dispatch <core/ParallelDispatch.html>`__
     - Parallel Execution Dispatch.
   * - `Built-in Reducers <core/builtin_reducers.html>`__
     - Built-in Reducers
   * - `Execution Policies <core/Execution-Policies.html>`__
     - Execution policies.
   * - `Spaces <core/Spaces.html>`__
     - Description of Memory and Execution Spaces.
   * - `Task-Parallelism <core/Task-Parallelism.html>`__
     - Creating and dispatching Task Graphs.
   * - `Atomics <core/atomics.html>`__
     - Atomics
   * - `Numerics <core/Numerics.html>`__
     - Common mathematical functions, mathematical constants, numeric traits.
   * - `C-style memory management <core/c_style_memory_management.html>`__
     - C-style memory management
   * - `Traits <core/Prod.html>`__
     - Traits
   * - `Kokkos Concepts <core/KokkosConcepts.html>`__
     - Kokkos Concepts
   * - `STL Compatibility Issues <core/STL-Compatibility.html>`__
     - Ports of standard C++ capabilities, which otherwise do not work on various hardware platforms.
   * - `Utilities <core/Utilities.html>`__
     - Utility functionality part of Kokkos Core.
   * - `Detection Idiom <core/Detection-Idiom.html>`__
     - Used to recognize, in an SFINAE-friendly way, the validity of any C++ expression.
   * - `Macros <core/Macros.html>`__
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
