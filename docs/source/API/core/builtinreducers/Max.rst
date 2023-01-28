``Max``
=======

.. role::cpp(code)
    :language: cpp

.. role:: cppkokkos(code)
    :language: cppkokkos

Specific implementation of `ReducerConcept <ReducerConcept.html>`_ storing the maximum value

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

    T result;
    parallel_reduce(N,Functor,Max<T,S>(result));

Synopsis
--------

.. code-block:: cpp

    template<class Scalar, class Space>
    class Max{
        public:
            typedef Max reducer;
            typedef typename std::remove_cv<Scalar>::type value_type;
            typedef Kokkos::View<value_type, Space> result_view_type;
            
            KOKKOS_INLINE_FUNCTION
            void join(value_type& dest, const value_type& src) const;

            KOKKOS_INLINE_FUNCTION
            void init(value_type& val) const;

            KOKKOS_INLINE_FUNCTION
            value_type& reference() const;

            KOKKOS_INLINE_FUNCTION
            result_view_type view() const;

            KOKKOS_INLINE_FUNCTION
            Max(value_type& value_);

            KOKKOS_INLINE_FUNCTION
            Max(const result_view_type& value_);
    };

Public Class Members
--------------------

Typedefs
~~~~~~~~
   
* ``reducer``: The self type.
* ``value_type``: The reduction scalar type.
* ``result_view_type``: A ``Kokkos::View`` referencing the reduction result 

Constructors
~~~~~~~~~~~~
 
.. cppkokkos:kokkosinlinefunction:: Max(value_type& value_);

    * Constructs a reducer which references a local variable as its result location.

.. cppkokkos:kokkosinlinefunction:: Max(const result_view_type& value_);

    * Constructs a reducer which references a specific view as its result location.

Functions
~~~~~~~~~

.. cppkokkos:kokkosinlinefunction:: void join(value_type& dest, const value_type& src) const;

    * Store maximum of ``src`` and ``dest`` into ``dest``: ``dest = ( src > dest ) ? src :dest;``. 

.. cppkokkos:kokkosinlinefunction:: void init(value_type& val) const;

    * Initialize ``val`` using the ``Kokkos::reduction_identity<Scalar>::max()`` method. The default implementation sets ``val=<TYPE>_MIN``.

.. cppkokkos:kokkosinlinefunction:: value_type& reference() const;

    * Returns a reference to the result provided in class constructor.

.. cppkokkos:kokkosinlinefunction:: result_view_type view() const;

    * Returns a view of the result place provided in class constructor.

Additional Information
~~~~~~~~~~~~~~~~~~~~~~

* ``Max<T,S>::value_type`` is non-const ``T``
* ``Max<T,S>::result_view_type`` is ``Kokkos::View<T,S,Kokkos::MemoryTraits<Kokkos::Unmanaged>>``. Note that the S (memory space) must be the same as the space where the result resides.
* Requires: ``Scalar` has ``operator =`` and ``operator >`` defined. ``Kokkos::reduction_identity<Scalar>::max()`` is a valid expression. 
* In order to use Max with a custom type, a template specialization of ``Kokkos::reduction_identity<CustomType>`` must be defined.  See `Built-In Reducers with Custom Scalar Types <../../../ProgrammingGuide/Custom-Reductions-Built-In-Reducers-with-Custom-Scalar-Types.html>`_ for details
