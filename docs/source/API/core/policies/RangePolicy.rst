``RangePolicy``
===============

.. role::cppkokkos(code)
    :language: cppkokkos

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cppkokkos

    Kokkos::RangePolicy<>(begin, end)
    Kokkos::RangePolicy<ARGS>(begin, end)
    Kokkos::RangePolicy<>(begin, end, chunk_size)
    Kokkos::RangePolicy<ARGS>(begin, end, chunk_size)
    Kokkos::RangePolicy<>(Space(), begin, end)
    Kokkos::RangePolicy<ARGS>(Space(), begin, end)
    Kokkos::RangePolicy<>(Space(), begin, end, chunk_size)
    Kokkos::RangePolicy<ARGS>(Space(), begin, end, chunk_size)

RangePolicy defines an execution policy for a 1D iteration space starting at begin and going to end with an open interval.

Synopsis
--------

.. code-block:: cpp

    struct Kokkos::ChunkSize {
        ChunkSize(int value_);
    };

    template<class ... Args>
    class Kokkos::RangePolicy {
        typedef RangePolicy execution_policy;
        typedef typename traits::index_type member_type;
        typedef typename traits::index_type index_type;

        //Inherited from PolicyTraits<Args...>
        using execution_space   = PolicyTraits<Args...>::execution_space;
        using schedule_type     = PolicyTraits<Args...>::schedule_type;
        using work_tag          = PolicyTraits<Args...>::work_tag;
        using index_type        = PolicyTraits<Args...>::index_type;
        using iteration_pattern = PolicyTraits<Args...>::iteration_pattern;
        using launch_bounds     = PolicyTraits<Args...>::launch_bounds;

        //Constructors
        RangePolicy(const RangePolicy&) = default;
        RangePolicy(RangePolicy&&) = default;

        RangePolicy();

        // since 4.3
        RangePolicy( member_type work_begin
                   , member_type work_end );

        // since 4.3
        RangePolicy( member_type work_begin
                   , member_type work_end
                   , ChunkSize chunk_size );

        // since 4.3
        RangePolicy( const execution_space & work_space
                   , member_type work_begin
                   , member_type work_end );

        // since 4.3
        RangePolicy( const execution_space & work_space
                   , member_type work_begin
                   , member_type work_end
                   , ChunkSize chunk_size );

        // until 4.3
        template<class ... Args>
        RangePolicy( const execution_space & work_space
                   , member_type work_begin
                   , member_type work_end
                   , Args ... args );

        // until 4.3
        template<class ... Args>
        RangePolicy( member_type work_begin
                   , member_type work_end
                   , Args ... args );

        // retrieve chunk_size
        member_type chunk_size() const;
        // set chunk_size to a discrete value
        RangePolicy& set_chunk_size(int chunk_size_);

        // return ExecSpace instance provided to the constructor
        KOKKOS_INLINE_FUNCTION const execution_space & space() const;
        // return Range begin
        KOKKOS_INLINE_FUNCTION member_type begin() const;
        // return Range end
        KOKKOS_INLINE_FUNCTION member_type end()   const;
    };

Parameters
----------

Common Arguments for all Execution Policies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Execution Policies generally accept compile time arguments via template parameters and runtime parameters via constructor arguments or setter functions.

* Template arguments can be given in arbitrary order.

+-------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
| Argument          | Options                                                                   | Purpose                                                                                                                                                 |
+===================+===========================================================================+=========================================================================================================================================================+
| ExecutionSpace    | ``Serial``, ``OpenMP``, ``Threads``, ``Cuda``, ``HIP``, ``SYCL``, ``HPX`` | Specify the Execution Space to execute the kernel in. Defaults to ``Kokkos::DefaultExecutionSpace``.                                                    |
+-------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
| Schedule          | ``Schedule<Dynamic>``, ``Schedule<Static>``                               | Specify scheduling policy for work items. ``Dynamic`` scheduling is implemented through a work stealing queue. Default is machine and backend specific. |
+-------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
| IndexType         | ``IndexType<int>``                                                        | Specify integer type to be used for traversing the iteration space. Defaults to ``int64_t``.                                                            |
+-------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
| LaunchBounds      | ``LaunchBounds<MaxThreads, MinBlocks>``                                   | Specifies hints to to the compiler about CUDA/HIP launch bounds.                                                                                        |
+-------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
| WorkTag           | ``SomeClass``                                                             | Specify the work tag type used to call the functor operator. Any arbitrary type defaults to ``void``.                                                   |
+-------------------+---------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------+

Public Class Members
--------------------

Constructors
~~~~~~~~~~~~

.. cppkokkos:function:: ChunkSize(int value_)

   Provide a hint for optimal chunk-size to be used during scheduling.
   For the SYCL backend, the workgroup size used in a ``parallel_for`` kernel can be set via this passed to ``RangePolicy``.

.. cppkokkos:function:: RangePolicy()

   Default Constructor uninitialized policy.

Since 4.3:
^^^^^^^^^^

.. cppkokkos:function:: RangePolicy(int64_t begin, int64_t end)

   Provide a start and end index.

.. cppkokkos:function:: RangePolicy(int64_t begin, int64_t end, ChunkSize chunk_size)

   Provide a start and end index as well as a ``ChunkSize``.

.. cppkokkos:function:: RangePolicy(const ExecutionSpace& space, int64_t begin, int64_t end)

   Provide a start and end index and an ``ExecutionSpace`` instance to use as the execution resource.

.. cppkokkos:function:: RangePolicy(const ExecutionSpace& space, int64_t begin, int64_t end, ChunkSize chunk_size)

   Provide a start and end index and an ``ExecutionSpace`` instance to use as the execution resource, as well as a ``ChunkSize``.

Until 4.3:
^^^^^^^^^^

.. cppkokkos:function:: template<class ... InitArgs> RangePolicy(int64_t begin, int64_t end, InitArgs ... init_args)

   Provide a start and end index as well as optional arguments to control certain behavior (see below).

.. cppkokkos:function:: template<class ... InitArgs> RangePolicy(const ExecutionSpace& space, int64_t begin, int64_t end, InitArgs ... init_args)

   Provide a start and end index and an ``ExecutionSpace`` instance to use as the execution resource, as well as optional arguments to control certain behavior (see below).

Optional ``InitArgs`` (until 4.3):
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``ChunkSize``

Preconditions:
^^^^^^^^^^^^^^

* The start index must not be greater than the end index.

CTAD Constructors (since 4.3):
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cppkokkos

   int64_t work_begin = /* ... */; // conversions as well
   int64_t work_end   = /* ... */; // conversions as well
   ChunkSize cs       = /* ... */; // conversions as well
   DefaultExecutionSpace des;      // conversions as well
   SomeExecutionSpace ses;         // different from DefaultExecutionSpace

   // Deduces to RangePolicy<>
   RangePolicy rp0;
   RangePolicy rp1(work_begin, work_end);
   RangePolicy rp2(work_begin, work_end, cs);
   RangePolicy rp3(des, work_begin, work_end);
   RangePolicy rp4(des, work_begin, work_end, cs);

   // Deduces to RangePolicy<SomeExecutionSpace>
   RangePolicy rp5(ses, work_begin, work_end);
   RangePolicy rp6(ses, work_begin, work_end, cs);

Examples
--------

.. code-block:: cppkokkos

    RangePolicy<> policy_1(0, N);
    RangePolicy<Cuda> policy_2(5,N-5);
    RangePolicy<Schedule<Dynamic>, OpenMP> policy_3(n,m);
    RangePolicy<IndexType<int>, Schedule<Dynamic>> policy_4(0, K);
    RangePolicy<> policy_6(-3,N+3, ChunkSize(8));
    RangePolicy<OpenMP> policy_7(OpenMP(), 0, N, ChunkSize(4));

Note: providing a single integer as a policy to a parallel pattern, implies a defaulted ``RangePolicy``

.. code-block:: cppkokkos

    // These two calls are identical
    parallel_for("Loop", N, functor);
    parallel_for("Loop", RangePolicy<>(0, N), functor);
