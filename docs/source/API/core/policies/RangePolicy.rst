``RangePolicy``
===============

.. role::cppkokkos(code)
    :language: cppkokkos

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cppkokkos

    Kokkos::RangePolicy<>(begin, end, args...)
    Kokkos::RangePolicy<ARGS>(begin, end, args...)
    Kokkos::RangePolicy<>(Space(), begin, end, args...)
    Kokkos::RangePolicy<ARGS>(Space(), begin, end, args...)

RangePolicy defines an execution policy for a 1D iteration space starting at begin and going to end with an open interval. 

Synopsis 
--------

.. code-block:: cppkokkos
        
    template<class ... Args>
    class Kokkos::RangePolicy {
        typedef RangePolicy execution_policy;
        typedef typename traits::index_type member_type ;
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

        inline RangePolicy();

        template<class ... Args>
        inline RangePolicy( const execution_space & work_space
                          , const member_type work_begin
                          , const member_type work_end
                          , Args ... args);

        template<class ... Args>
        inline RangePolicy( const member_type work_begin
                          , const member_type work_end
                          , Args ... args);

        // retrieve chunk_size
        inline member_type chunk_size() const;
        // set chunk_size to a discrete value
        inline RangePolicy set_chunk_size(int chunk_size_);

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

Requirements
~~~~~~~~~~~~

Public Class Members
--------------------

Constructors
~~~~~~~~~~~~
 
.. code-block:: cppkokkos
    
    RangePolicy()

\
    Default Constructor uninitialized policy.

.. code-block:: cppkokkos

    template<class ... InitArgs> 
    RangePolicy(const int64_t& begin, const int64_t& end, const InitArgs ... init_args)

\
    Provide a start and end index as well as optional arguments to control certain behavior (see below).
   
.. code-block:: cppkokkos

    template<class ... InitArgs> 
    RangePolicy(const ExecutionSpace& space, const int64_t& begin, const int64_t& end, const InitArgs ... init_args)

\
    Provide a start and end index and an ``ExecutionSpace`` instance to use as the execution resource, as well as optional arguments to control certain behavior (see below).

Optional ``InitArgs``:
^^^^^^^^^^^^^^^^^^^^^^

* ``ChunkSize`` : Provide a hint for optimal chunk-size to be used during scheduling. For the SYCL backend, the workgroup size used in a ``parallel_for`` kernel can be set via this variable. 

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
