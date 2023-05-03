
``for_each``
============

Header: ``<Kokkos_StdAlgorithms.hpp>``

Description
-----------

Applies a unary functor to the result of dereferencing each iterator in a range or each element in a rank-1 ``View``.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.

.. code-block:: cpp

   //
   // overload set accepting an execution space
   //
   template <class ExecutionSpace, class InputIterator, class UnaryFunctorType>
   UnaryFunctorType for_each(const ExecutionSpace& exespace,                            (1)
                             InputIterator first, InputIterator last,
			     UnaryFunctorType func);

   template <class ExecutionSpace, class InputIterator, class UnaryFunctorType>
   UnaryFunctorType for_each(const std::string& label, const ExecutionSpace& exespace,  (2)
			     InputIterator first, InputIterator last,
			     UnaryFunctorType func);

   template <class ExecutionSpace, class DataType, class... Properties, class UnaryFunctorType>
   UnaryFunctorType for_each(const ExecutionSpace& exespace,                            (3)
		             const Kokkos::View<DataType, Properties...>& view,
                             UnaryFunctorType func);

   template <class ExecutionSpace, class DataType, class... Properties, class PredicateType>
   UnaryFunctorType for_each(const std::string& label, const ExecutionSpace& exespace,  (4)
		             const Kokkos::View<DataType, Properties...>& view,
			     UnaryFunctorType func);

   //
   // overload set accepting a team handle
   //
   template <class TeamHandleType, class InputIterator, class UnaryFunctorType>
   KOKKOS_FUNCTION
   UnaryFunctorType for_each(const TeamHandleType& teamHandle,                          (5)
			     InputIterator first, InputIterator last,
			     UnaryFunctorType func);

   template <class TeamHandleType, class DataType, class... Properties, class UnaryFunctorType>
   KOKKOS_FUNCTION
   UnaryFunctorType for_each(const TeamHandleType& teamHandle,                          (6)
                             const Kokkos::View<DataType, Properties...>& view,
			     UnaryFunctorType func);

Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``exespace``: execution space instance

- ``teamHandle``: team handle instance given inside a parallel region when using a TeamPolicy

- ``label``: string forwarded to internal parallel kernels for debugging purposes

  - for 1, the default string is: "Kokkos::for_each_iterator_api_default"

  - for 3, the default string is: "Kokkos::for_each_view_api_default"

  - NOTE: overloads accepting a team handle do not use a label internally

- ``first, last``: iterators defining the ranges to operate on

  - must be *random access iterators*

  - must represent a valid range, i.e., ``last >= first``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``view``:

  - must be rank-1, and have ``LayoutLeft``, ``LayoutRight``, or ``LayoutStride``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``func``: function object called on the all the elements;

  - The signature of the function should be ``func(v)`` and must be valid to be called from the execution space passed,
    or the execution space associated with the team handle, and must accept every argument ``v`` of type
    ``value_type``, where ``value_type`` is the value type of ``InputIterator`` or ``view``

  - must conform to:

    .. code-block:: cpp

       struct func
       {
	  KOKKOS_INLINE_FUNCTION
	  void operator()(const /*type needed */ & operand) const { /* ... */; }

	  // or, also valid

	  KOKKOS_INLINE_FUNCTION
	  void operator()(/*type needed */ & operand) const { /* ... */; }
       };

Return
~~~~~~

``func``

Example
-------

.. code-block:: cpp

   namespace KE = Kokkos::Experimental;

   template<class ValueType>
   struct IncrementValFunctor
   {
     const ValueType m_value;
     IncrementValFunctor(ValueType value) : m_value(value){}

     KOKKOS_INLINE_FUNCTION
     void operator()(ValueType & operand) const {
       operand += m_value;
     }
   };

   auto exespace = Kokkos::DefaultExecutionSpace;
   using view_type = Kokkos::View<exespace, int*>;
   view_type a("a", 15);
   // fill "a" somehow

   // create functor
   IncrementValFunctor<int> p(5);

   // Increment each element in "a" by 5.
   KE::for_each(exespace, KE::begin(a), KE::end(a), p);

   // assuming OpenMP is enabled, then you can also explicitly call
   KE::for_each(Kokkos::OpenMP(), KE::begin(a), KE::end(a), p);
