``MDRangePolicy``
=================

.. role:: cppkokkos(code)
    :language: cppkokkos

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

    Kokkos::MDRangePolicy<>(begin, end)
    Kokkos::MDRangePolicy<>(Space, begin, end)
    Kokkos::MDRangePolicy<ARGS>(begin, end, tiling)
    Kokkos::MDRangePolicy<ARGS>(Space, begin, end, tiling)

``MDRangePolicy`` defines an execution policy for a multidimensional iteration space starting at a ``begin`` tuple and going to ``end`` with an open interval. The iteration space will be tiled, and the user can optionally provide tiling sizes.

Interface
---------

.. code-block:: cpp

    template<class ... Args>
    class Kokkos::MDRangePolicy;

Parameters
----------

Common Arguments for all Execution Policies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Execution Policies generally accept compile time arguments via template parameters and runtime parameters via constructor arguments or setter functions.
* Template arguments can be given in arbitrary order.

+----------------+----------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
| Argument       | Options                                                                    | Purpose                                                                                                                                                 |
+================+============================================================================+=========================================================================================================================================================+
| ExecutionSpace |  ``Serial``, ``OpenMP``, ``Threads``, ``Cuda``, ``HIP``, ``SYCL``, ``HPX`` | Specify the Execution Space to execute the kernel in. Defaults to ``Kokkos::DefaultExecutionSpace``.                                                    |
+----------------+----------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
| Schedule       | ``Schedule<Dynamic>``, ``Schedule<Static>``                                | Specify scheduling policy for work items. ``Dynamic`` scheduling is implemented through a work stealing queue. Default is machine and backend specific. |
+----------------+----------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
| IndexType      | ``IndexType<int>``                                                         | Specify integer type to be used for traversing the iteration space. Defaults to ``int64_t``.                                                            |
+----------------+----------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
| LaunchBounds   | ``LaunchBounds<MaxThreads, MinBlocks>``                                    | Specifies hints to to the compiler about CUDA/HIP launch bounds.                                                                                        |
+----------------+----------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
| WorkTag        | ``SomeClass``                                                              | Specify the work tag type used to call the functor operator. Any arbitrary type defaults to ``void``.                                                   |
+----------------+----------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------+

Arguments Specific to MDRangePolicy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

    template<int N, Iterate outer, Iterate inner>
    class Rank;

* Determines the rank of the index space as well as in which order to iterate over the tiles and how to iterate within the tiles. ``outer`` and ``inner`` can be ``Kokkos::Iterate::Default``, ``Kokkos::Iterate::Left``, or ``Kokkos::Iterate::Right``.


Public Class Members
--------------------

Constructors
~~~~~~~~~~~~

.. cppkokkos:function:: MDRangePolicy()

    * Default Constructor uninitialized policy.

.. cppkokkos:function:: MDRangePolicy(const Kokkos::Array<int64_t,rank>& begin, const Kokkos::Array<int64_t,rank>& end)

    * Provide a start and end index.

.. cppkokkos:function:: MDRangePolicy(const Kokkos::Array<int64_t,rank>& begin, const Kokkos::Array<int64_t,rank>& end,  const Kokkos::Array<int64_t,rank>& tiling)

    * Provide a start and end index as well as the tiling dimensions.

.. cppkokkos:function:: template<class OT, class IT, class TT> MDRangePolicy(const std::initializer_list<OT>& begin, const std::initializer_list<IT>& end)

    * Provide a start and end index. The length of the lists must match the rank of the policy.

.. cppkokkos:function:: template<class OT, class IT, class TT> MDRangePolicy(const std::initializer_list<OT>& begin, const std::initializer_list<IT>& end,  std::initializer_list<TT>& tiling)

    * Provide a start and end index as well as the tiling dimensions. The length of the lists must match the rank of the policy.

Preconditions:

* The start index must not be greater than the matching end index for all ranks.

Examples
--------

.. code-block:: cpp

    MDRangePolicy<Rank<3>> policy_1({0,0,0},{N0,N1,N2});
    MDRangePolicy<Cuda,Rank<3,Iterate::Right,Iterate::Left>> policy_2({5,5,5},{N0-5,N1-5,N2-5},{T0,T1,T2});
