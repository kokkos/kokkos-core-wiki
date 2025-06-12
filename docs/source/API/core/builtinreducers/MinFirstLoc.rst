``MinFirstLoc``
===============

.. role:: cpp(code)
    :language: cpp

Specific implementation of `ReducerConcept <ReducerConcept.html>`_ storing the minimum value.
If there are equivalent values, stores the smallest index.

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

   MinFirstLoc<T, I, S>::value_type result;
   parallel_reduce(N, Functor, MinFirstLoc<T, I, S>(result));

Synopsis
--------

.. code-block:: cpp

   template<typename Scalar, typename Index, typename Space = HostSpace>
   struct MinFirstLoc{
       using reducer = MinFirstLoc;
       using value_type = ValLocScalar<std::remove_cv_t<Scalar>, std::remove_cv_t<Index>>;
       using result_view_type = View<value_type, Space>;

       MinFirstLoc(value_type& value_);
       MinFirstLoc(const result_view_type& value_);

       void join(value_type& dest, const value_type& src) const;
       void init(value_type& val) const;
       value_type& reference() const;
       result_view_type view() const;
       bool references_scalar() const;
   };

   template<typename T, typename I, typename... Ps>
   MinFirstLoc(View<ValLocScalar<T, I>, Ps...> const&)
   -> MinFirstLoc<T, I, View<ValLocScalar<T, I>, Ps...>::memory_space>;

Interface
---------

.. cpp:class:: template<class Scalar, class Index, class Space> MinFirstLoc

   .. rubric:: Public Types:

   .. cpp:type:: reducer

      The self type.

   .. cpp:type:: value_type

      The reduction scalar type (specialization of `ValLocScalar <ValLocScalar.html>`_).

   .. cpp:type:: result_view_type

      A ``View`` referencing the reduction result.

   .. rubric:: Constructors:

   .. cpp:function:: MinFirstLoc(value_type& value_);

      Constructs a reducer which references a local variable as its result location.

   .. cpp:function:: MinFirstLoc(const result_view_type& value_);

      Constructs a reducer which references a specific view as its result location.

   .. rubric:: Public Member Functions:

   .. cpp:function:: void join(value_type& dest, const value_type& src) const;

      If ``src.val == dest.val`` && ``src.loc < dest.loc`` then ``dest.loc = src.loc``;
      otherwise if ``src.val < dest.val`` then ``dest.val = src.val`` && ``dest.loc = src.loc``.

   .. cpp:function:: void init(value_type& val) const;

      Initialize ``val.val`` using the ``reduction_identity<Scalar>::min()`` method. The default implementation sets ``val=<TYPE>_MAX``.
      Initialize ``val.loc`` using the ``reduction_identity<Index>::min()`` method. The default implementation sets ``val=<TYPE>_MAX``.

   .. cpp:function:: value_type& reference() const;

      :return: A reference to the result provided in class constructor.

   .. cpp:function:: result_view_type view() const;

      :return: A ``View`` of the result place provided in class constructor.

   .. cpp:function:: bool references_scalar() const;

      :return: ``true`` if the reducer was constructed with a scalar; ``false`` if the reducer was constructed with a ``View``.

   .. rubric:: Explicit Deduction Guides (CTAD):

   .. cpp:function:: template<typename T, typename I, typename... Ps> MinFirstLoc(View<ValLocScalar<T, I>, Ps...> const&) -> MinFirstLoc<T, I, View<ValLocScalar<T, I>, Ps...>::memory_space>;

Additional Information
^^^^^^^^^^^^^^^^^^^^^^

* ``MinFirstLoc<T, I, S>::value_type`` is specialization of ``ValLocScalar`` on non-``const`` ``T`` and non-``const`` ``I``.

* ``MinFirstLoc<T, I, S>::result_view_type`` is ``View<T,S,MemoryTraits<Unmanaged>>``. Note that the ``S`` (memory space) must be the same as the space where the result resides.

* Requires: ``Scalar`` has ``operator=`` and ``operator>`` defined. ``reduction_identity<Scalar>::max()`` is a valid expression.

* Requires: ``Index`` has ``operator=`` defined. ``reduction_identity<Index>::min()`` is a valid expression.

* In order to use ``MinFirstLoc`` with a custom type of either ``Scalar`` or ``Index``, a template specialization of ``reduction_identity<CustomType>`` must be defined. See `Built-In Reducers with Custom Scalar Types <../../../ProgrammingGuide/Custom-Reductions-Built-In-Reducers-with-Custom-Scalar-Types.html>`_ for details.
