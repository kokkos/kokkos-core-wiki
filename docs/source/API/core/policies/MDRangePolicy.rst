``MDRangePolicy``
=================

.. role:: cppkokkos(code)
    :language: cppkokkos

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

    Kokkos::MDRangePolicy<..., Rank<N>, ...>(begin, end)
    Kokkos::MDRangePolicy<..., Rank<N>, ...>(Space, begin, end)
    Kokkos::MDRangePolicy<..., Rank<N>, ...>(begin, end, tiling)
    Kokkos::MDRangePolicy<..., Rank<N>, ...>(Space, begin, end, tiling)

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

Required Arguments Specific to MDRangePolicy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

    template<int N,
             Kokkos::Iterate outer = Kokkos::Iterate::Default,
             Kokkos::Iterate inner = Kokkos::Iterate::Default>
    class Kokkos::Rank;

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

CTAD Constructors (since 4.3):
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cppkokkos

   DefaultExecutionSpace des;
   SomeExecutionSpace ses; // different from DefaultExecutionSpace
   int64_t i;

   // Deduces to MDRangePolicy<Rank<5>>
   MDRangePolicy pl0({1, 2, 3, 4, 5}, {1, 2, 3, 4, 5});
   MDRangePolicy pl1({1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}, { i });
   MDRangePolicy pl2(des, {1, 2, 3, 4, 5}, {1, 2, 3, 4, 5});
   MDRangePolicy pl3(des, {1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}, { i });

   // Deduces to MDRangePolicy<SomeExecutionSpace, Rank<5>>
   MDRangePolicy pl4(ses, {1, 2, 3, 4, 5}, {1, 2, 3, 4, 5});
   MDRangePolicy pl5(ses, {1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}, { i });

   int cbegin[3];
   int cend[3];
   int64_t ctiling[2];

   // Deduces to MDRangePolicy<Rank<3>>
   MDRangePolicy pc0(cbegin, cend);
   MDRangePolicy pc1(cbegin, cend, ctiling);
   MDRangePolicy pc2(des, cbegin, cend);
   MDRangePolicy pc3(des, cbegin, cend, ctiling);

   // Deduces to MDRangePolicy<SomeExecutionSpace, Rank<3>>
   MDRangePolicy pc4(ses, cbegin, cend);
   MDRangePolicy pc5(ses, cbegin, cend, ctiling);

   Array<int, 2> abegin;
   Array<int, 2> aend;
   Array<int, 1> atiling;

   // Deduces to MDRangePolicy<Rank<2>>
   MDRangePolicy pa0(abegin, aend);
   MDRangePolicy pa1(abegin, aend, atiling);
   MDRangePolicy pa2(des, abegin, aend);
   MDRangePolicy pa3(des, abegin, aend, atiling);

   // Deduces to MDRangePolicy<SomeExecutionSpace, Rank<2>>
   MDRangePolicy pa4(ses, abegin, aend);
   MDRangePolicy pa5(ses, abegin, aend, atiling);

Preconditions:
^^^^^^^^^^^^^^

* The start index must not be greater than the matching end index for all ranks.
* The begin & end array ranks must match.
* The tiling array rank must be less than or equal to the begin/end array rank.

Examples
--------

.. code-block:: cpp

    MDRangePolicy<Rank<3>> policy_1({0,0,0},{N0,N1,N2});
    MDRangePolicy<Cuda,Rank<3,Iterate::Right,Iterate::Left>> policy_2({5,5,5},{N0-5,N1-5,N2-5},{T0,T1,T2});
