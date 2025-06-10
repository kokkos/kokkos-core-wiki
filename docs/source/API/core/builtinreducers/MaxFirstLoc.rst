``MaxFirstLoc``
===============

.. role:: cpp(code)
    :language: cpp

Specific implementation of `ReducerConcept <ReducerConcept.html>`_ storing the maximum value.
If there are equivalent values, stores the smallest index.

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

   MaxFirstLoc<T,I,S>::value_type result;
   parallel_reduce(N,Functor,MaxFirstLoc<T,I,S>(result));

Synopsis
--------

.. code-block:: cpp

   template<typename Scalar, typename Index, typename Space>
   struct MaxFirstLoc{
       using reducer = MaxFirstLoc;
       using value_type = ValLocScalar<std::remove_cv_t<Scalar>, std::remove_cv_t<Index>>;
       using result_view_type = View<value_type, Space>;

       MaxFirstLoc(value_type& value_);
       MaxFirstLoc(const result_view_type& value_);

       void join(value_type& dest, const value_type& src) const;
       void init(value_type& val) const;
       value_type& reference() const;
       result_view_type view() const;
       bool references_scalar() const;
   };

Interface
---------

.. cpp:class:: template<class Scalar, class Index, class Space> MaxFirstLoc

   .. rubric:: Public Types

   .. cpp:type:: reducer

      The self type.

   .. cpp:type:: value_type

      The reduction scalar type (specialization of `ValLocScalar <ValLocScalar.html>`_).

   .. cpp:type:: result_view_type

      A ``View`` referencing the reduction result.

   .. rubric:: Constructors

   .. cpp:function:: MaxFirstLoc(value_type& value_);

      Constructs a reducer which references a local variable as its result location.

   .. cpp:function:: MaxFirstLoc(const result_view_type& value_);

      Constructs a reducer which references a specific view as its result location.

   .. rubric:: Public Member Functions

   .. cpp:function:: void join(value_type& dest, const value_type& src) const;

      Store maximum with index of ``dest`` and ``src`` into ``dest``. If ``dest.val == src.val``, the location stored is ``std::min(dest.loc, src.loc)`` (the first one found).

   .. cpp:function:: void init(value_type& val) const;

      Initialize ``val.val`` using the ``reduction_identity<Scalar>::max()`` method. The default implementation sets ``val=<TYPE>_MIN``.
      Initialize ``val.loc`` using the ``reduction_identity<Index>::min()`` method. The default implementation sets ``val=<TYPE>_MAX``.

   .. cpp:function:: value_type& reference() const;

      Returns a reference to the result provided in class constructor.

   .. cpp:function:: result_view_type view() const;

      Returns a view of the result place provided in class constructor.

   .. cpp:function:: bool references_scalar() const;

      :return: ``true`` if the reducer was constructed with a scalar; ``false`` if the reducer was constructed with a ``View``.

Additional Information
^^^^^^^^^^^^^^^^^^^^^^

* ``MaxFirstLoc<T,I,S>::value_type`` is Specialization of ValLocScalar on non-const ``T`` and non-const ``I``

* ``MaxFirstLoc<T,I,S>::result_view_type`` is ``View<T,S,MemoryTraits<Unmanaged>>``. Note that the S (memory space) must be the same as the space where the result resides.

* Requires: ``Scalar`` has ``operator =`` and ``operator >`` defined. ``reduction_identity<Scalar>::max()`` is a valid expression.

* Requires: ``Index`` has ``operator =`` defined. ``reduction_identity<Index>::min()`` is a valid expression.

* In order to use MaxFirstLoc with a custom type of either ``Scalar`` or ``Index``, a template specialization of ``reduction_identity<CustomType>`` must be defined. See `Built-In Reducers with Custom Scalar Types <../../../ProgrammingGuide/Custom-Reductions-Built-In-Reducers-with-Custom-Scalar-Types.html>`_ for details
