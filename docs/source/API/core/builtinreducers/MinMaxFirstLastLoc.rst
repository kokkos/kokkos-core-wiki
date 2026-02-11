``MinMaxFirstLastLoc``
======================

.. role:: cpp(code)
    :language: cpp

Specific implementation of `ReducerConcept <ReducerConcept.html>`_ storing both the minimum and maximum values with corresponding indices.  If there are equivalent minimum values, stores the smallest corresponding index.  If there are equivalent maximum values, stores the largest corresponding index.

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

   MinMaxFirstLastLoc<T, I, S>::value_type result;
   parallel_reduce(N, Functor, MinMaxFirstLastLoc<T, I, S>(result));

Synopsis
--------

.. code-block:: cpp

   template<typename Scalar, typename Index, typename Space = HostSpace>
   struct MinMaxFirstLastLoc {
       using reducer = MinMaxFirstLastLoc;
       using value_type = MinMaxLocScalar<std::remove_cv_t<Scalar>, std::remove_cv_t<Index>>;
       using result_view_type = View<value_type, Space>;

       MinMaxFirstLastLoc(value_type& value_);
       MinMaxFirstLastLoc(const result_view_type& value_);

       void join(value_type& dest, const value_type& src) const;
       void init(value_type& val) const;
       value_type& reference() const;
       result_view_type view() const;
       bool references_scalar() const;
   };

   template<typename T, typename I, typename... Ps>
   MinMaxFirstLastLoc(View<MinMaxLocScalar<T, I>, Ps...> const&)
   -> MinMaxFirstLastLoc<T, I, View<MinMaxLocScalar<T, I>, Ps...>::memory_space>;

Interface
---------

.. cpp:class:: template<typename Scalar, typename Index, typename Space> MinMaxFirstLastLoc

   .. rubric:: Public Types:

   .. cpp:type:: reducer

      The self type.

   .. cpp:type:: value_type

      The reduction scalar type (specialization of `MinMaxLocScalar <MinMaxLocScalar.html>`_)

   .. cpp:type:: result_view_type

      A ``View`` referencing the reduction result.

   .. rubric:: Constructors:

   .. cpp:function:: MinMaxFirstLastLoc(value_type& value_);

      Constructs a reducer which references a local variable as its result location.

   .. cpp:function:: MinMaxFirstLastLoc(const result_view_type& value_);

      Constructs a reducer which references a specific view as its result location.

   .. rubric:: Public Member Functions:

   .. cpp:function:: void join(value_type& dest, const value_type& src) const;

      If ``dest.min_val == src.min_val && src.min_loc < dest.min_loc`` then ``dest.min_loc = src.min_loc``;
      otherwise if ``src.min_val < dest.min_val`` then ``dest.min_val = src.min_val`` & ``dest.min_loc = src.min_loc``.

      If ``dest.max_val == src.max_val && src.max_loc > dest.max_loc`` then ``dest.max_loc = src.max_loc``.
      otherwise if ``dest.max_val < src.max_val`` then ``dest.max_val = src.max_val`` & ``dest.max_loc = src.max_loc``.

   .. cpp:function:: void init(value_type& val) const;

      Initialize ``val.min_val`` using the ``Kokkos::reduction_identity<Scalar>::min()`` method. The default implementation sets ``val=<TYPE>_MAX``.

      Initialize ``val.max_val`` using the ``Kokkos::reduction_identity<Index>::max()`` method. The default implementation sets ``val=<TYPE>_MIN``.

      Initialize ``val.min_loc`` using the ``Kokkos::reduction_identity<Scalar>::min()`` method. The default implementation sets ``val=<TYPE>_MAX``.

      Initialize ``val.max_loc`` using the ``Kokkos::reduction_identity<Index>::max()`` method. The default implementation sets ``val=<TYPE>_MAX``.

   .. cpp:function:: value_type& reference() const;

      :return: A reference to the result provided in class constructor.

   .. cpp:function:: result_view_type view() const;

      :return: A ``View`` of the result place provided in class constructor.

   .. cpp:function:: bool references_scalar() const;

      :return: ``true`` if the reducer was constructed with a scalar; ``false`` if the reducer was constructed with a ``View``.

   .. rubric:: Explicit Deduction Guides (CTAD):

   .. cpp:function:: template<typename T, typename I, typename... Ps> MinMaxFirstLastLoc(View<MinMaxLocScalar<T, I>, Ps...> const&) -> MinMaxFirstLastLoc<T, I, View<MinMaxLocScalar<T, I>, Ps...>::memory_space>;

Additional Information
^^^^^^^^^^^^^^^^^^^^^^

* ``MinMaxFirstLastLoc<T, I, S>::value_type`` is specialization of ``MinMaxLocScalar`` on non-``const`` ``T`` and non-``const`` ``I``.

* ``MinMaxFirstLastLoc<T, I, S>::result_view_type`` is ``View<T, S, MemoryTraits<Unmanaged>>``. Note that the ``S`` (memory space) must be the same as the space where the result resides.

* Requires: ``Scalar`` has ``operator=``, ``operator<`` and ``operator>`` defined. ``reduction_identity<Scalar>::min()`` & ``reduction_identity<Scalar>::max()`` are valid expressions.

* Requires: ``Index`` has ``operator=`` defined. ``reduction_identity<Index>::min()`` is a valid expression.

* In order to use ``MinMaxFirstLastLoc`` with a custom type of either ``Scalar`` or ``Index``, a template specialization of ``reduction_identity<CustomType>`` must be defined. See `Built-In Reducers with Custom Scalar Types <../../../ProgrammingGuide/Custom-Reductions-Built-In-Reducers-with-Custom-Scalar-Types.html>`_ for details.

