Execution Policies
##################

Top Level Execution Policies
============================

:doc:`ExecutionPolicyConcept <policies/ExecutionPolicyConcept>` is the fundamental abstraction to represent “how” the execution of a Kokkos parallel pattern takes place.

.. list-table::
    :widths: 35 65
    :header-rows: 1
    :align: left

    * - Policy
      - Description

    * * :doc:`RangePolicy <policies/RangePolicy>`
      * Each iterate is an integer in a contiguous range

    * * :doc:`MDRangePolicy <policies/MDRangePolicy>`
      * Each iterate for each rank is an integer in a contiguous range

    * * :doc:`TeamPolicy <policies/TeamPolicy>`
      * Assigns to each iterate in a contiguous range a team of threads

Nested Execution Policies
============================

Nested Execution Policies are used to dispatch parallel work inside of an already executing parallel region either dispatched with a :doc:`TeamPolicy <policies/TeamPolicy>` or a task policy. :doc:`NestedPolicies <policies/NestedPolicies>` summary.

.. list-table::
    :widths: 25 75
    :header-rows: 1
    :align: left

    * - Policy
      - Description

    * * :doc:`TeamThreadMDRange <policies/TeamThreadMDRange>`
      * Used inside of a TeamPolicy kernel to perform nested parallel loops over a multidimensional range split over threads of a team.

    * * :doc:`TeamThreadRange <policies/TeamThreadRange>`
      * Used inside of a TeamPolicy kernel to perform nested parallel loops split over threads of a team.

    * * :doc:`TeamVectorMDRange <policies/TeamVectorMDRange>`
      * Used inside of a TeamPolicy kernel to perform nested parallel loops over a multidimensional range split over threads of a team and their vector lanes.

    * * :doc:`TeamVectorRange <policies/TeamVectorRange>`
      * Used inside of a TeamPolicy kernel to perform nested parallel loops split over threads of a team and their vector lanes.

    * * :doc:`ThreadVectorMDRange <policies/ThreadVectorMDRange>`
      * Used inside of a TeamPolicy kernel to perform nested parallel loops over a multidimensional range with vector lanes of a thread.

    * * :doc:`ThreadVectorRange <policies/ThreadVectorRange>`
      * Used inside of a TeamPolicy kernel to perform nested parallel loops with vector lanes of a thread.

.. _kokkos-common-arguments-for-all-execution-policies:

Common Arguments for all Execution Policies
===========================================

Execution Policies generally accept compile time arguments via template parameters and runtime parameters via constructor arguments or setter functions.

.. tip::

    Template arguments can be given in arbitrary order.

.. list-table::
    :widths: 30 30 40
    :header-rows: 1
    :align: left

    * - Argument
      - Options
      - Purpose

    * * ExecutionSpace
      * ``Serial``, ``OpenMP``, ``Threads``, ``Cuda``, ``HIP``, ``SYCL``, ``HPX``
      * Specify the Execution Space to execute the kernel in. Defaults to ``Kokkos::DefaultExecutionSpace``

    * * Schedule
      * ``Schedule<Dynamic>``, ``Schedule<Static>``
      * Specify scheduling policy for work items. ``Dynamic`` scheduling is implemented through a work stealing queue. Default is machine and backend specific.

    * * IndexType
      * e.g. ``IndexType<int>``
      * Specify integer type to be used for traversing the iteration space. Defaults to the ``size_type`` of :doc:`ExecutionSpaceConcept <execution_spaces>`. Can affect the performance depending on the backend.

    * * LaunchBounds
      * ``LaunchBounds<MaxThreads, MinBlocks>``
      * Specifies hints to to the compiler about CUDA/HIP launch bounds.

    * * WorkTag
      * ``SomeClass``
      * Specify the work tag type used to call the functor operator. Can be any arbitrary tag type (i.e. an [empty](https://en.cppreference.com/w/cpp/types/is_empty) struct or class). Defaults to ``void``.


.. toctree::
   :hidden:
   :maxdepth: 1

   ./policies/ExecutionPolicyConcept
   ./policies/MDRangePolicy
   ./policies/NestedPolicies
   ./policies/RangePolicy
   ./policies/TeamHandleConcept
   ./policies/TeamPolicy
   ./policies/TeamThreadMDRange
   ./policies/TeamThreadRange
   ./policies/TeamVectorMDRange
   ./policies/TeamVectorRange
   ./policies/ThreadVectorMDRange
   ./policies/ThreadVectorRange
