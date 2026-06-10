``BAnd``
========

.. role:: cpp(code)
    :language: cpp

Specific implementation of `ReducerConcept <ReducerConcept.html>`_ performing bitwise ``AND`` operation

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

    T result;
    parallel_reduce(N, Functor, BAnd<T, S>(result));

Synopsis
--------

.. code-block:: cpp

    template<class Scalar, class Space>
    class BAnd {
      public:
        using reducer = BAnd<Scalar, Space>;
        using value_type = typename std::remove_cv<Scalar>::type;

        KOKKOS_INLINE_FUNCTION
        void join(value_type& dest, const value_type& src) const {
          dest = dest & src;
        }

        KOKKOS_INLINE_FUNCTION
        void init(value_type& val) const {
          val = Kokkos::reduction_identity<value_type>::band();
        }
    };

Interface
---------

All the public types, constructors and methods from `ReducerConcept <ReducerConcept.html>`_ are available. The following types and methods are overridden by this reducer:

.. cpp:class:: template<class Scalar, class Space> BAnd

   .. rubric:: Public Types

   .. cpp:type:: reducer

      The self type.

   .. cpp:type:: value_type

      The non-const version of the ``Scalar`` template parameter.      

   .. rubric:: Public Member Functions

   .. cpp:function:: KOKKOS_INLINE_FUNCTION void join(value_type& dest, const value_type& src) const;

      Store bitwise ``and`` of ``src`` and ``dest`` into ``dest``:  ``dest = src & dest``.

   .. cpp:function:: KOKKOS_INLINE_FUNCTION void init(value_type& val) const;

      Initialize ``val`` using the ``Kokkos::reduction_identity<value_type>::band()`` method. The default implementation sets ``val=~(0x0)``.


Additional Information
~~~~~~~~~~~~~~~~~~~~~~

* Requires: ``Scalar`` has ``operator =`` and ``operator &`` defined. ``Kokkos::reduction_identity<value_type>::band()`` is a valid expression.

* In order to use ``BAnd`` with a custom type, a template specialization of ``Kokkos::reduction_identity<CustomType>`` must be defined. See `Built-In Reducers with Custom Scalar Types <../../../ProgrammingGuide/Custom-Reductions-Built-In-Reducers-with-Custom-Scalar-Types.html>`_ for details
