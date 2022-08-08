Parallel Execution/Dispatch
===========================

Parallel patterns
-----------------

Parallel execution patterns for composing algorithms.

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Function
     - Description
   * - `parallel_for <parallel-dispatch/parallel_for.html>`__
     - Executes user code in parallel
   * - `parallel_reduce <parallel-dispatch/parallel_reduce.html>`__
     - Executes user code to perform a reduction in parallel
   * - `parallel_scan <parallel-dispatch/parallel_scan.html>`__
     - Executes user code to generate a prefix sum in parallel
   * - `fence <parallel-dispatch/fence.html>`__
     - Fences execution spaces

Tags for Team Policy Calculations
---------------------------------

The following parallel pattern tags are used to call the correct overload for team size calculations (team_size_max,team_size_recommended):

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Tag
     - Pattern
   * - `ParallelForTag <parallel-dispatch/ParallelForTag.html>`__
     - parallel_for
   * - `ParallelReduceTag <parallel-dispatch/ParallelReduceTag.html>`__
     - parallel_reduce
   * - `ParallelScanTag <parallel-dispatch/ParallelScanTag.html>`__
     - parallel_scan

.. toctree::
   :hidden:
   :maxdepth: 1

   ./parallel-dispatch/parallel_for
   ./parallel-dispatch/parallel_reduce
   ./parallel-dispatch/parallel_scan
   ./parallel-dispatch/fence
   ./parallel-dispatch/ParallelForTag
   ./parallel-dispatch/ParallelReduceTag
   ./parallel-dispatch/ParallelScanTag
