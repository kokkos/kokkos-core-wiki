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
   * - :doc:`parallel_for <parallel-dispatch/parallel_for>`
     - Executes user code in parallel
   * - :doc:`parallel_reduce <parallel-dispatch/parallel_reduce>`
     - Executes user code to perform a reduction in parallel
   * - :doc:`parallel_scan <parallel-dispatch/parallel_scan>`
     - Executes user code to generate a prefix sum in parallel
   * - :doc:`fence <parallel-dispatch/fence>`
     - Fences execution spaces

Tags for Team Policy Calculations
---------------------------------

The following parallel pattern tags are used to call the correct overload for team size calculations (team_size_max,team_size_recommended):

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Tag
     - Pattern
   * - :doc:`ParallelForTag <parallel-dispatch/ParallelForTag>`
     - parallel_for
   * - :doc:`ParallelReduceTag <parallel-dispatch/ParallelReduceTag>`
     - parallel_reduce
   * - :doc:`ParallelScanTag <parallel-dispatch/ParallelScanTag>`
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
