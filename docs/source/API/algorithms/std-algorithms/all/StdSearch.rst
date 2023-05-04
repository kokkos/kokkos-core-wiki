
``search``
==========

Header: ``<Kokkos_StdAlgorithms.hpp>``

Description
-----------

Searches in a range or a ``View`` for the first occurrence of a target sequence of elements.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.

.. code-block:: cpp

   //
   // overload set accepting an execution space
   //
   template <class ExecutionSpace, class IteratorType1, class IteratorType2>
   IteratorType1 search(const ExecutionSpace& exespace,                           (1)
			IteratorType1 first, IteratorType1 last,
			IteratorType2 s_first, IteratorType2 s_last);

   template <class ExecutionSpace, class IteratorType1, class IteratorType2>
   IteratorType1 search(const std::string& label, const ExecutionSpace& exespace, (2)
			IteratorType1 first, IteratorType1 last,
			IteratorType2 s_first, IteratorType2 s_last);

   template <class ExecutionSpace, class DataType1, class... Properties1,
	     class DataType2, class... Properties2>
   auto search(const ExecutionSpace& exespace,                                    (3)
	       const ::Kokkos::View<DataType1, Properties1...>& view,
	       const ::Kokkos::View<DataType2, Properties2...>& s_view);

   template <class ExecutionSpace, class DataType1, class... Properties1,
	     class DataType2, class... Properties2>
   auto search(const std::string& label, const ExecutionSpace& exespace,          (4)
	       const ::Kokkos::View<DataType1, Properties1...>& view,
	       const ::Kokkos::View<DataType2, Properties2...>& s_view);

   // overload set 2: binary predicate passed
   template <class ExecutionSpace, class IteratorType1, class IteratorType2,
	     class BinaryPredicateType>
   IteratorType1 search(const ExecutionSpace& exespace,                           (5)
                        IteratorType1 first, IteratorType1 last,
			IteratorType2 s_first, IteratorType2 s_last,
			const BinaryPredicateType& pred);

   template <class ExecutionSpace, class IteratorType1, class IteratorType2,
	     class BinaryPredicateType>
   IteratorType1 search(const std::string& label, const ExecutionSpace& exespace, (6)
			IteratorType1 first, IteratorType1 last,
			IteratorType2 s_first, IteratorType2 s_last,
			const BinaryPredicateType& pred);

   template <class ExecutionSpace, class DataType1, class... Properties1,
	     class DataType2, class... Properties2, class BinaryPredicateType>
   auto search(const ExecutionSpace& exespace,                                    (7)
	       const ::Kokkos::View<DataType1, Properties1...>& view,
	       const ::Kokkos::View<DataType2, Properties2...>& s_view,
	       const BinaryPredicateType& pred);

   template <class ExecutionSpace, class DataType1, class... Properties1,
	     class DataType2, class... Properties2, class BinaryPredicateType>
   auto search(const std::string& label, const ExecutionSpace& exespace,          (8)
	       const ::Kokkos::View<DataType1, Properties1...>& view,
	       const ::Kokkos::View<DataType2, Properties2...>& s_view,
	       const BinaryPredicateType& pred)

   //
   // overload set accepting a team handle
   //
   template <class TeamHandleType, class IteratorType1, class IteratorType2>
   KOKKOS_FUNCTION
   IteratorType1 search(const TeamHandleType& teamHandle,                         (9)
			IteratorType1 first, IteratorType1 last,
			IteratorType2 s_first, IteratorType2 s_last);

   template <class TeamHandleType, class DataType1, class... Properties1,
	     class DataType2, class... Properties2>
   KOKKOS_FUNCTION
   auto search(const TeamHandleType& teamHandle,                                 (10)
	       const ::Kokkos::View<DataType1, Properties1...>& view,
	       const ::Kokkos::View<DataType2, Properties2...>& s_view);

   // overload set 2: binary predicate passed
   template <class TeamHandleType, class IteratorType1, class IteratorType2,
	     class BinaryPredicateType>
   KOKKOS_FUNCTION
   IteratorType1 search(const TeamHandleType& teamHandle,                        (11)
                        IteratorType1 first, IteratorType1 last,
			IteratorType2 s_first, IteratorType2 s_last,
			const BinaryPredicateType& pred);

   template <class TeamHandleType, class DataType1, class... Properties1,
	     class DataType2, class... Properties2, class BinaryPredicateType>
   KOKKOS_FUNCTION
   auto search(const TeamHandleType& teamHandle,                                 (12)
	       const ::Kokkos::View<DataType1, Properties1...>& view,
	       const ::Kokkos::View<DataType2, Properties2...>& s_view,
	       const BinaryPredicateType& pred);


Detailed Description
~~~~~~~~~~~~~~~~~~~~

- 1,2,5,6,9,11: Searches for the first occurrence of the sequence of elements ``[s_first, s_last)`` in the range ``[first, last)``

- 3,4,7,8,10,12: Searches for the first occurrence of the sequence of elements ``s_view`` in ``view``

Elements are compared using ``pred`` (where accepted), otherwise via ``operator ==``.


Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``exespace``: execution space instance

- ``teamHandle``: team handle instance given inside a parallel region when using a TeamPolicy

- ``label``: string forwarded to internal parallel kernels for debugging purposes

  - 1,5: The default string is "Kokkos::search_iterator_api_default".

  - 3,7: The default string is "Kokkos::search_view_api_default".

  - NOTE: overloads accepting a team handle do not use a label internally

- ``first, last``: range of elements to search in

  - must be *random access iterators*, e.g., returned from ``Kokkos::Experimental::(c)begin/(c)end``

  - must represent a valid range, i.e., ``last >= first``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``s_first, s_last``: range of elements that you want to search for

  - same requirements as ``first, last``

- ``view``, ``s_view``: views to search in and for, respectively

  - must be rank-1, and have ``LayoutLeft``, ``LayoutRight``, or ``LayoutStride``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``pred``: *binary* functor returning ``true`` if two arguments should be considered "equal".

  ``pred(a,b)`` must be valid to be called from the execution space passed, or
  the execution space associated with the team handle, and convertible to bool
  for every pair of arguments ``a,b`` of type ``ValueType1`` and ``ValueType2``,
  respectively, where ``ValueType1`` and ``ValueType{1,2}`` are the value types of
  ``IteratorType{1,2}`` or ``(s_)view``, and must not modify ``a,b``.

  - must conform to:

  .. code-block:: cpp

     template <class ValueType1, class ValueType2 = ValueType1>
     struct IsEqualFunctor {
      KOKKOS_INLINE_FUNCTION
      bool operator()(const ValueType1& a, const ValueType2& b) const {
        return (a == b);
      }
     };


Return Value
~~~~~~~~~~~~

- for overloads accepting iterators: returns a ``IteratorType1`` instance pointing to the beginning
  of the sequence ``[s_first, s_last)`` in the range ``[first, last)``, or ``last`` if no such element is found.
  If the sequence ``[s_first, s_last)`` is empty, ``first`` is returend.

- for overloads accepting Views: returns a Kokkos iterator to the first element in ``view`` that markes the beginning of ``s_view``
  or ``Kokkos::Experimental::end(view)`` if no such element is found.
  If the sequence ``[s_first, s_last)`` is empty, ``Kokkos::Experimental::begin(view)`` is returend.
