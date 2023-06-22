``RangePolicy``
===============

.. role::cppkokkos(code)
    :language: cppkokkos

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cppkokkos

    Kokkos::RangePolicy<...>(begin, end)
    Kokkos::RangePolicy<...>(begin, end, ChunkSize(cs))
    Kokkos::RangePolicy<...>(Space(), begin, end)
    Kokkos::RangePolicy<...>(Space(), begin, end, ChunkSize(cs))

RangePolicy defines an execution policy for a 1D iteration space starting at ``begin`` and going to ``end`` with an open interval.

Synopsis
--------

.. code-block:: cpp

   template<
            typename ExecutionSpace = DefaultExecutionSpace
          , typename S
          , typename IT = int64_t
          , unsigned MaxThreads
          , unsigned MinBlocks,
          , typename WorkTag = void>
    class RangePolicy<
                      ExecutionSpace
                    , Schedule<S>
                    , IndexType<IT>
                    , LaunchBounds<MaxThreads, MinBlocks>
                    , WorkTag> {
        using execution_policy = RangePolicy;
        using execution_space  = ExecutionSpace;
        using schedule_type    = Schedule<S>;
        using index_type       = IT;
        using member_type      = IT;
        using launch_bounds    = LaunchBounds<MaxThreads, MinBlocks>;
        using work_tag         = WorkTag;

        // Constructors
        RangePolicy();

        RangePolicy(index_type work_begin
                  , index_type work_end);

        RangePolicy(index_type work_begin
                  , index_type work_end
                  , ChunkSize cs);

        RangePolicy(const execution_space& work_space
                  , index_type work_begin
                  , index_type work_end)

        RangePolicy(const execution_space& work_space
                  , index_type work_begin
                  , index_type work_end
                  , ChunkSize cs);

        // Getters and setters
        index_type chunk_size() const;
        RangePolicy set_chunk_size(int chunk_size_);

        const ExecutionSpace& space() const;

        index_type begin() const;
        index_type end()   const;
    };

Parameters
----------

Common Arguments for all Execution Policies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Execution Policies generally accept compile time arguments via template parameters and runtime parameters via constructor arguments or setter functions.

* Template arguments are all optional and can be given in arbitrary order.

+-------------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------+----------------------------+
| Argument          | Options                                                                   | Purpose                                                                                                        | Default                    |
+===================+===========================================================================+================================================================================================================+============================+
| ExecutionSpace    | ``Serial``, ``OpenMP``, ``Threads``, ``Cuda``, ``HIP``, ``SYCL``, ``HPX`` | Specify the Execution Space to execute the kernel in.                                                          | ``DefaultExecutionSpace``  |
+-------------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------+----------------------------+
| Schedule          | ``Schedule<Dynamic>``, ``Schedule<Static>``                               | Specify scheduling policy for work items. ``Dynamic`` scheduling is implemented through a work stealing queue. | machine / backend specific |
+-------------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------+----------------------------+
| IndexType         | ``IndexType<IT>``                                                         | Specify integer type to be used for traversing the iteration space.                                            | ``IndexType<int64_t>``     |
+-------------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------+----------------------------+
| LaunchBounds      | ``LaunchBounds<MaxThreads, MinBlocks>``                                   | Specifies hints to to the compiler about CUDA/HIP launch bounds.                                               |                            |
+-------------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------+----------------------------+
| WorkTag           | ``SomeClass``                                                             | Specify the work tag type used to call the functor operator.                                                   | ``void``                   |
+-------------------+---------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------+----------------------------+

Public Class Members
--------------------

Constructors
~~~~~~~~~~~~

.. cppkokkos:function:: RangePolicy()

   Construct an uninitialized policy.

.. cppkokkos:function:: RangePolicy(index_type begin, index_type end)

   Construct a policy with a start index and an end index.

.. cppkokkos:function:: RangePolicy(index_type begin, index_type end, ChunkSize cs)

   Construct a policy with a start index, an end index and a ChunkSize (see below).

.. cppkokkos:function:: RangePolicy(const execution_space& space, index_type begin, index_type end)

   Construct a policy with an execution space instance to be used as the execution resource, a start index, and an end index.

.. cppkokkos:function:: RangePolicy(const execution_space& space, index_type begin, index_type end, ChunkSize cs)

   Construct a policy with an execution space instance to be used as the execution resource, a start index, an end index and a ChunkSize (see below).

* ``ChunkSize`` : Provide a hint for optimal chunk-size to be used during scheduling. For the SYCL backend, the workgroup size used in a ``parallel_for`` kernel can be set via this variable.

Examples
--------

.. code-block:: cppkokkos

    RangePolicy<>                                  policy_1(0, N);
    RangePolicy<Cuda>                              policy_2(5, N-5);
    RangePolicy<Schedule<Dynamic>, OpenMP>         policy_3(n, m);
    RangePolicy<IndexType<int>, Schedule<Dynamic>> policy_4(0, K);
    RangePolicy<>                                  policy_5(-3, N+3, ChunkSize(8));
    RangePolicy<OpenMP>                            policy_6(OpenMP(), 0, N, ChunkSize(4));

Note: providing a single integer as a policy to a parallel pattern implies a defaulted ``RangePolicy``.

.. code-block:: cppkokkos

    // These two calls act identically
    parallel_for("Loop", N, functor);
    parallel_for("Loop", RangePolicy<>(0, N), functor);
