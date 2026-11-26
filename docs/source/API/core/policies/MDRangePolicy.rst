``MDRangePolicy``
=================

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

    Kokkos::MDRangePolicy<..., Rank<N>, ...>(lower, upper)
    Kokkos::MDRangePolicy<..., Rank<N>, ...>(Space, lower, upper)
    Kokkos::MDRangePolicy<..., Rank<N>, ...>(lower, upper, tiling)
    Kokkos::MDRangePolicy<..., Rank<N>, ...>(Space, lower, upper, tiling)

    // CTAD Constructors (since 4.3)
    Kokkos::MDRangePolicy(lower, upper)
    Kokkos::MDRangePolicy(Space, lower, upper)
    Kokkos::MDRangePolicy(lower, upper, tiling)
    Kokkos::MDRangePolicy(Space, lower, upper, tiling)

``MDRangePolicy`` defines an execution policy for a multidimensional iteration space starting at the ``lower`` bounds (inclusive) and ending at the ``upper`` bounds (exclusive). The iteration space is tiled, and the user can optionally provide tile sizes.

Interface
---------

.. cpp:class:: template<class ... Args> Kokkos::MDRangePolicy

Parameters
----------

General Template Arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~

Valid template arguments for ``MDRangePolicy`` are described `here <../Execution-Policies.html#common-arguments-for-all-execution-policies>`_.

Required Argument Specific to MDRangePolicy - ``Kokkos::Rank``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rank Interface
^^^^^^^^^^^^^^
.. code-block:: cpp

    template<int N,
             Kokkos::Iterate outer = Kokkos::Iterate::Default,
             Kokkos::Iterate inner = Kokkos::Iterate::Default>
    class Kokkos::Rank;

``Kokkos::Rank`` is a template argument unique to ``MDRangePolicy``. It can be deduced using CTAD (since 4.3).
It specifies the rank of the iteration space and, optionally, the iteration order over all the tiles and within each tile.

``outer`` and ``inner`` default to ``Kokkos::Iterate::Default`` and can be set to ``Kokkos::Iterate::Left`` or ``Kokkos::Iterate::Right``.

Template Arguments
^^^^^^^^^^^^^^^^^^

.. cpp:class:: template<int N, Kokkos::Iterate outer, Kokkos::Iterate inner> Kokkos::Rank

  :tparam N: Rank of the iteration space (1 to 6).
  :tparam outer: Iteration order over tiles (optional).
  :tparam inner: Iteration order within each tile (optional).

  .. note:: Rank 1 is supported since Kokkos 5.2.

.. cpp:enum-class:: Kokkos::Iterate

    .. cpp:enumerator:: Default

        Use the natural iteration order for the execution space.

    .. cpp:enumerator:: Left

        Column-major: leftmost index varies fastest.

    .. cpp:enumerator:: Right

        Row-major: rightmost index varies fastest.

    .. note::

        For best performance, match the iteration order to your View's memory layout. See :ref:`Iteration Order <MDRangePolicy-Iteration-order>` in the Programming Guide.

Public Class Members
--------------------

.. cpp:namespace-push:: MDRangePolicy

Nested typedefs
~~~~~~~~~~~~~~~

.. cpp:type:: execution_space
.. cpp:type:: iteration_pattern

    Alias to ``Kokkos::Rank<...>`` used by the policy itself.

.. cpp:type:: work_tag
.. cpp:type:: index_type
.. cpp:type:: array_index_type

    Alias to ``std::int64_t``.

.. cpp:type:: launch_bounds
.. cpp:type:: point_type

    Alias to ``Kokkos::Array<index_type, rank>``.

    .. versionchanged:: 5.3

       Previously ``Kokkos::Array<array_index_type, rank>``.

.. cpp:type:: tile_type

    Alias to ``Kokkos::Array<index_type, rank>``.

    .. versionchanged:: 5.3

       Previously ``Kokkos::Array<array_index_type, rank>``.


Constructors
~~~~~~~~~~~~

.. cpp:function:: MDRangePolicy()

    * Default constructor. Creates an uninitialized policy.

.. cpp:function:: MDRangePolicy(const Kokkos::Array<int64_t,rank>& lower, const Kokkos::Array<int64_t,rank>& upper)
                  template<class OT, class IT> MDRangePolicy(const std::initializer_list<OT>& lower, const std::initializer_list<IT>& upper)

    * Provide the lower and upper bounds. When given as initializer lists, their length must match the rank of the policy.

.. cpp:function:: MDRangePolicy(const execution_space& space, const Kokkos::Array<int64_t,rank>& lower, const Kokkos::Array<int64_t,rank>& upper)
                  template<class OT, class IT> MDRangePolicy(const execution_space& space, const std::initializer_list<OT>& lower, const std::initializer_list<IT>& upper)

    * Provide the lower and upper bounds, and use the provided execution space instance. When given as initializer lists, their length must match the rank of the policy.

.. cpp:function:: MDRangePolicy(const Kokkos::Array<int64_t,rank>& lower, const Kokkos::Array<int64_t,rank>& upper, const Kokkos::Array<int64_t,rank>& tiling)
                  template<class OT, class IT, class TT> MDRangePolicy(const std::initializer_list<OT>& lower, const std::initializer_list<IT>& upper, const std::initializer_list<TT>& tiling)

    * Provide the lower and upper bounds as well as the tile sizes. When given as initializer lists, their length must match the rank of the policy.

.. cpp:function:: MDRangePolicy(const execution_space& space, const Kokkos::Array<int64_t,rank>& lower, const Kokkos::Array<int64_t,rank>& upper, const Kokkos::Array<int64_t,rank>& tiling)
                  template<class OT, class IT, class TT> MDRangePolicy(const execution_space& space, const std::initializer_list<OT>& lower, const std::initializer_list<IT>& upper, const std::initializer_list<TT>& tiling)

    * Provide the lower and upper bounds as well as the tile sizes, and use the provided execution space instance. When given as initializer lists, their length must match the rank of the policy.

CTAD Constructors (since 4.3)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

   DefaultExecutionSpace des;
   SomeExecutionSpace ses; // different from DefaultExecutionSpace

   // Deduces to MDRangePolicy<Rank<3>>
   MDRangePolicy pl0({0, 0, 0}, {4, 5, 10});
   MDRangePolicy pl1({0, 0, 0}, {4, 5, 10}, {3, 3, 3});

   // Deduces to MDRangePolicy<SomeExecutionSpace, Rank<3>>
   MDRangePolicy pl2(ses, {0, 0, 0}, {4, 5, 10});
   MDRangePolicy pl3(ses, {0, 0, 0}, {4, 5, 10}, {3, 3, 3});

   int clower_bounds[3];
   int cupper_bounds[3];
   int64_t ctiling[3];

   // Deduces to MDRangePolicy<Rank<3>>
   MDRangePolicy pc0(clower_bounds, cupper_bounds);
   MDRangePolicy pc1(clower_bounds, cupper_bounds, ctiling);
   MDRangePolicy pc2(des, clower_bounds, cupper_bounds);
   MDRangePolicy pc3(des, clower_bounds, cupper_bounds, ctiling);

   // Deduces to MDRangePolicy<SomeExecutionSpace, Rank<3>>
   MDRangePolicy pc4(ses, clower_bounds, cupper_bounds);
   MDRangePolicy pc5(ses, clower_bounds, cupper_bounds, ctiling);

   Array<int, 2> alower_bounds;
   Array<int, 2> aupper_bounds;
   Array<int, 2> atiling;

   // Deduces to MDRangePolicy<Rank<2>>
   MDRangePolicy pa0(alower_bounds, aupper_bounds);
   MDRangePolicy pa1(alower_bounds, aupper_bounds, atiling);
   MDRangePolicy pa2(des, alower_bounds, aupper_bounds);
   MDRangePolicy pa3(des, alower_bounds, aupper_bounds, atiling);

   // Deduces to MDRangePolicy<SomeExecutionSpace, Rank<2>>
   MDRangePolicy pa4(ses, alower_bounds, aupper_bounds);
   MDRangePolicy pa5(ses, alower_bounds, aupper_bounds, atiling);

Member Functions
~~~~~~~~~~~~~~~~
.. cpp:function:: const execution_space& space() const

    * Returns a const reference to the underlying ``execution_space`` instance. If no execution space instance was provided to the constructor, the underlying instance is the default execution space instance.

.. cpp:function:: point_type lower() const

    * Returns the inclusive lower bounds of the multidimensional range.

    .. versionadded:: 5.3

.. cpp:function:: point_type upper() const

    * Returns the exclusive upper bounds of the multidimensional range.

    .. versionadded:: 5.3

.. cpp:function:: tile_type tile_size() const

    * Returns the tile size used by the policy.

    .. versionadded:: 5.3

.. cpp:function:: tile_type tile_size_recommended() const

    * Returns a ``tile_type`` containing per-rank tile sizes that are internally recommended to an instance of ``MDRangePolicy``. The default tile sizes are static and are set based on the specified backend.

    .. note:: ``tile_size_recommended()`` available since Kokkos 4.5

.. cpp:function:: int max_total_tile_size() const

    * Returns the upper limit for the product of all tile sizes.

    .. note:: ``max_total_tile_size()`` available since Kokkos 4.5

.. cpp:namespace-pop::

Notes
~~~~~

* For each rank, the lower bound must not be greater than the matching upper bound.
* The lower and upper arrays must have the same length.
* The tiling array length must be less than or equal to the lower/upper array length.

Examples
--------

.. code-block:: cpp

    MDRangePolicy<> policy_1({0,0,0}, {N0,N1,N2});
    MDRangePolicy<Rank<3>> policy_2({5,5,5}, {N0-5,N1-5,N2-5}, {T0,T1,T2});
    MDRangePolicy<Cuda, Rank<3, Iterate::Right, Iterate::Left>> policy_3({0,0,0}, {N0,N1,N2});

.. code-block:: cpp

    MDRangePolicy<Rank<2>> md_policy({0,0}, {10,10});

    parallel_for("fill", md_policy, KOKKOS_LAMBDA(int i, int j) {
        a(i, j) = i + j;
    });

    double sum;
    parallel_reduce("sum", md_policy, KOKKOS_LAMBDA(int i, int j, double& update) {
        update += a(i, j);
    }, sum);
