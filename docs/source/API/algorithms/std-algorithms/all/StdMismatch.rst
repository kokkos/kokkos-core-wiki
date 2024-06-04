
``mismatch``
============

Header: ``<Kokkos_StdAlgorithms.hpp>``

Description
-----------

Returns the first mismatching pair of elements from two ranges or two views.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.

Overload set accepting execution space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <class ExecutionSpace, class IteratorType1, class IteratorType2>
   Kokkos::pair<IteratorType1, IteratorType2> mismatch(const ExecutionSpace& exespace,  (1)
                                                       IteratorType1 first1,
						       IteratorType1 last1,
						       IteratorType2 first2,
						       IteratorType2 last2);

   template <class ExecutionSpace, class IteratorType1, class IteratorType2>
   Kokkos::pair<IteratorType1, IteratorType2> mismatch(const std::string& label,        (2)
						       const ExecutionSpace& exespace,
						       IteratorType1 first1,
						       IteratorType1 last1,
						       IteratorType2 first2,
						       IteratorType2 last2)

   template <class ExecutionSpace, class IteratorType1, class IteratorType2, class BinaryPredicate>
   Kokkos::pair<IteratorType1, IteratorType2> mismatch(const ExecutionSpace& exespace,  (3)
						       IteratorType1 first1,
						       IteratorType1 last1,
						       IteratorType2 first2,
						       IteratorType2 last2,
						       BinaryPredicate pred);

   template <class ExecutionSpace, class IteratorType1, class IteratorType2, class BinaryPredicate>
   Kokkos::pair<IteratorType1, IteratorType2> mismatch(const std::string& label,        (4)
						       const ExecutionSpace& exespace,
						       IteratorType1 first1,
						       IteratorType1 last1,
						       IteratorType2 first2,
						       IteratorType2 last2,
						       BinaryPredicate pred);

   template <class ExecutionSpace, class DataType1, class... Properties1,
	     class DataType2, class... Properties2>
   auto mismatch(const ExecutionSpace& exespace,
		 const Kokkos::View<DataType1, Properties1...>& view1,                  (5)
		 const Kokkos::View<DataType2, Properties2...>& view2);

   template <class ExecutionSpace, class DataType1, class... Properties1,
	     class DataType2, class... Properties2>
   auto mismatch(const std::string& label, const ExecutionSpace& exespace,
		 const Kokkos::View<DataType1, Properties1...>& view1,                  (6)
		 const Kokkos::View<DataType2, Properties2...>& view2);

   template <class ExecutionSpace, class DataType1, class... Properties1,
	     class DataType2, class... Properties2, class BinaryPredicateType>
   auto mismatch(const ExecutionSpace& exespace,
		 const Kokkos::View<DataType1, Properties1...>& view1,                  (7)
		 const Kokkos::View<DataType2, Properties2...>& view2,
		 BinaryPredicateType&& predicate);

   template <class ExecutionSpace, class DataType1, class... Properties1,
	     class DataType2, class... Properties2, class BinaryPredicateType>
   auto mismatch(const std::string& label, const ExecutionSpace& exespace,
		 const Kokkos::View<DataType1, Properties1...>& view1,                  (8)
		 const Kokkos::View<DataType2, Properties2...>& view2,
		 BinaryPredicateType&& predicate);

Overload set accepting a team handle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 4.2

.. code-block:: cpp

   template <class TeamHandleType, class IteratorType1, class IteratorType2>
   KOKKOS_FUNCTION
   Kokkos::pair<IteratorType1, IteratorType2> mismatch(const TeamHandleType& teamHandle,  (9)
                                                       IteratorType1 first1,
						       IteratorType1 last1,
						       IteratorType2 first2,
						       IteratorType2 last2);

   template <class TeamHandleType, class IteratorType1, class IteratorType2,
             class BinaryPredicate>
   KOKKOS_FUNCTION
   Kokkos::pair<IteratorType1, IteratorType2> mismatch(const TeamHandleType& teamHandle,  (10)
						       IteratorType1 first1,
						       IteratorType1 last1,
						       IteratorType2 first2,
						       IteratorType2 last2,
						       BinaryPredicate pred);

   template <class TeamHandleType, class DataType1, class... Properties1,
	     class DataType2, class... Properties2>
   KOKKOS_FUNCTION
   auto mismatch(const TeamHandleType& teamHandle,
		 const Kokkos::View<DataType1, Properties1...>& view1,                  (11)
		 const Kokkos::View<DataType2, Properties2...>& view2);

   template <class TeamHandleType, class DataType1, class... Properties1,
	     class DataType2, class... Properties2, class BinaryPredicateType>
   KOKKOS_FUNCTION
   auto mismatch(const TeamHandleType& teamHandle,
		 const Kokkos::View<DataType1, Properties1...>& view1,                  (12)
		 const Kokkos::View<DataType2, Properties2...>& view2,
		 BinaryPredicateType&& predicate);

Detailed Description
~~~~~~~~~~~~~~~~~~~~

- 1,2,3,4,9,10: Returns the first mismatching pair of elements from two ranges: one defined
  by ``[first1, last1)`` and another defined by ``[first2,last2)``

- 5,6,7,8,11,12: Returns the first mismatching pair of elements from the two views ``view1`` and ``view2``

Comparison of elements is done via the BinaryPredicate, ``pred``, where provided, otherwise
using ``operator==``.


Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``exespace``: execution space instance

- ``teamHandle``: team handle instance given inside a parallel region when using a TeamPolicy

- ``label``: string forwarded to internal parallel kernels for debugging purposes

  - for 1,3, the default string is: "Kokkos::mismatch_iterator_api_default"

  - for 5,7, the default string is: "Kokkos::mismatch_view_api_default"

  - NOTE: overloads accepting a team handle do not use a label internally

- ``first1``, ``last1``, ``first2``, ``last2``: range of elements to compare

  - must be *random access iterators*, e.g., returned from ``Kokkos::Experimental::(c)begin/(c)end``

  - must represent valid ranges, i.e., ``last1 >= first1`` and ``last2 >= first2``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``view1``, ``view2``: views to compare
  - must be rank-1, and have ``LayoutLeft``, ``LayoutRight``, or ``LayoutStride``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``pred``

  - must conform to:

  .. code-block:: cpp

     template <class ValueType1, class ValueType2 = ValueType1>
     struct IsEqualFunctor {

     KOKKOS_INLINE_FUNCTION
     Kokkos::pair<ValueType1, ValueType2> operator()(const ValueType1& a, const ValueType2& b) const {
       return (a == b);
       }
     };

Return Value
~~~~~~~~~~~~

- a Kokkos::pair, where the ``.first`` and ``.second`` are the iterator instances
  where the ``operator==`` evaluates to false, or ``pred`` evaluates to false

Example
~~~~~~~

.. code-block:: cpp

   namespace KE = Kokkos::Experimental;

   template <class ValueType1, class ValueType2 = ValueType1>
   struct MismatchFunctor {

     KOKKOS_INLINE_FUNCTION
     Kokkos::pair<ValueType1, ValueType2> operator()(const ValueType1& a, const ValueType2& b) const {
       if(a != b)
	   return (Kokkos::pair<ValueType1, ValueType2> (a,b));
     }
   };

   auto exespace = Kokkos::DefaultExecutionSpace;
   using view_type = Kokkos::View<exespace, int*>;
   view_type a("a", 15);
   view_type b("b", 15);
   // fill a,b somehow

   // create functor
   MisMatchFunctor<int, int> p();

   Kokkos::pair<int,int> mismatch_index = KE::mismatch(exespace, KE::begin(a), KE::end(a), KE::begin(b), KE::end(b) p);

   // assuming OpenMP is enabled, then you can also explicitly call
// To run explicitly on the Host, (assuming a and b are accessible on Host)
   Kokkos::pair<int,int> mismatch_index = KE::mismatch(Kokkos::DefaultHostExecutionSpace(), KE::begin(a), KE::end(a), KE::begin(b), KE::end(b), p);
