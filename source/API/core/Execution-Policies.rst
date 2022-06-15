Execution Policies
##################

Top Level Execution Policies
============================

.. list-table::
    :widths: 35 65
    :header-rows: 1
    :align: left

    * - Policy
      - Description

    * * `RangePolicy <tbd>`__ 
      * Each iterate is an integer in a contiguous range

    * * `MDRangePolicy <mdrangepolicy>`_
      * Each iterate for each rank is an integer in a contiguous range

    * * `TeamPolicy <tbd>`__
      * Assigns to each iterate in a contiguous range a team of threads

Nested Execution Policies
============================

Nested Execution Policies are used to dispatch parallel work inside of an already executing parallel region either dispatched with a `TeamPolicy <tbd>`__ or a task policy. 

.. list-table::
    :widths: 25 75
    :header-rows: 1
    :align: left

    * - Policy
      - Description

    * * `TeamThreadRange <tbd>`__ 
      * Used inside of a TeamPolicy kernel to perform nested parallel loops split over threads of a team.
    
    * * `TeamVectorRange <tbd>`__ 
      * Used inside of a TeamPolicy kernel to perform nested parallel loops split over threads of a team and their vector lanes.
    
    * * `ThreadVectorRange <tbd>`__ 
      * Used inside of a TeamPolicy kernel to perform nested parallel loops with vector lanes of a thread.

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
      * ``IndexType<int>`` 
      * Specify integer type to be used for traversing the iteration space. Defaults to ``int64_t``.

    * * LaunchBounds 
      * ``LaunchBounds<MaxThreads, MinBlocks>`` 
      * Specifies hints to to the compiler about CUDA/HIP launch bounds.

    * * WorkTag 
      * ``SomeClass`` 
      * Specify the work tag type used to call the functor operator. Any arbitrary type defaults to ``void``.


.. toctree::
   :maxdepth: 1

  ./policies/RangePolicy
  ./policies/MDRangePolicy
  ./policies/TeamPolicy
  ./policies/TeamThreadRange
  ./policies/TeamVectorRange
  ./policies/ThreadVectorRange

