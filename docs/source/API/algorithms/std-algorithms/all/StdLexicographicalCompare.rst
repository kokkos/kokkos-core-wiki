
``lexicographical_compare``
===========================

Header: ``<Kokkos_StdAlgorithms.hpp>``

Description
-----------

Returns ``true`` if the first range (or view) is lexicographically less than the second range (or view).

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.

Overload set accepting execution space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <class ExecutionSpace, class IteratorType1, class IteratorType2>
   bool lexicographical_compare(const ExecutionSpace& exespace, IteratorType1 first1,
				IteratorType1 last1, IteratorType2 first2,              (1)
				IteratorType2 last2);

   template <class ExecutionSpace, class IteratorType1, class IteratorType2>
   bool lexicographical_compare(const std::string& label, const ExecutionSpace& exespace,
				IteratorType1 first1, IteratorType1 last1,              (2)
				IteratorType2 first2, IteratorType2 last2);

   template <class ExecutionSpace, class DataType1, class... Properties1,
	     class DataType2, class... Properties2>
   bool lexicographical_compare(const ExecutionSpace& exespace,                         (3)
                                const ::Kokkos::View<DataType1, Properties1...>& view1,
				::Kokkos::View<DataType2, Properties2...>& view2);

   template <class ExecutionSpace, class DataType1, class... Properties1,
	     class DataType2, class... Properties2>
   bool lexicographical_compare(const std::string& label,                               (4)
				const ExecutionSpace& exespace,
				const ::Kokkos::View<DataType1, Properties1...>& view1,
				::Kokkos::View<DataType2, Properties2...>& view2);

   template <class ExecutionSpace, class IteratorType1, class IteratorType2,
	     class ComparatorType>
   bool lexicographical_compare(const ExecutionSpace& exespace, IteratorType1 first1,
				IteratorType1 last1, IteratorType2 first2,              (5)
				IteratorType2 last2, ComparatorType comp);

   template <class ExecutionSpace, class IteratorType1, class IteratorType2,
	     class ComparatorType>
   bool lexicographical_compare(const std::string& label, const ExecutionSpace& exespace,
				IteratorType1 first1, IteratorType1 last1,              (6)
				IteratorType2 first2, IteratorType2 last2,
				ComparatorType comp);

   template <class ExecutionSpace, class DataType1, class... Properties1,
	     class DataType2, class... Properties2, class ComparatorType>
   bool lexicographical_compare(const ExecutionSpace& exespace,                         (7)
                                const ::Kokkos::View<DataType1, Properties1...>& view1,
			        ::Kokkos::View<DataType2, Properties2...>& view2,
			        ComparatorType comp);


   template <class ExecutionSpace, class DataType1, class... Properties1,
	     class DataType2, class... Properties2, class ComparatorType>
   bool lexicographical_compare(const std::string& label,                               (8)
				const ExecutionSpace& exespace,
				const ::Kokkos::View<DataType1, Properties1...>& view1,
				::Kokkos::View<DataType2, Properties2...>& view2,
				ComparatorType comp);

Overload set accepting a team handle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 4.2

.. code-block:: cpp

   template <class TeamHandleType, class IteratorType1, class IteratorType2>
   KOKKOS_FUNCTION
   bool lexicographical_compare(const TeamHandleType& teamHandle, IteratorType1 first1,
				IteratorType1 last1, IteratorType2 first2,              (9)
				IteratorType2 last2);

   template <class TeamHandleType, class DataType1, class... Properties1,
	     class DataType2, class... Properties2>
   KOKKOS_FUNCTION
   bool lexicographical_compare(const TeamHandleType& teamHandle,                      (10)
                                const ::Kokkos::View<DataType1, Properties1...>& view1,
				::Kokkos::View<DataType2, Properties2...>& view2);

   template <class TeamHandleType, class IteratorType1, class IteratorType2,
	     class ComparatorType>
   KOKKOS_FUNCTION
   bool lexicographical_compare(const TeamHandleType& teamHandle, IteratorType1 first1,
				IteratorType1 last1, IteratorType2 first2,             (11)
				IteratorType2 last2, ComparatorType comp);

   template <class TeamHandleType, class DataType1, class... Properties1,
	     class DataType2, class... Properties2, class ComparatorType>
   KOKKOS_FUNCTION
   bool lexicographical_compare(const TeamHandleType& teamHandle,                      (12)
                                const ::Kokkos::View<DataType1, Properties1...>& view1,
			        ::Kokkos::View<DataType2, Properties2...>& view2,
			        ComparatorType comp);

Detailed Description
~~~~~~~~~~~~~~~~~~~~

Returns ``true`` for if the first range ``[first1, last1)`` (or ``view1``) is lexicographically
less than the second range ``[first2, last2)`` (or ``view2``).

Elements are compared using the ``<`` operator for all overloads not accepting a comparison object ``comp``.

Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``exespace``: execution space instance

- ``teamHandle``: team handle instance given inside a parallel region when using a TeamPolicy

- ``label``:

  - 1,5: The default string is "Kokkos::lexicographical_compare_iterator_api_defaul".

  - 3,7: The default string is "Kokkos::lexicographical_compare_view_api_default".

- ``first1``, ``last1``, ``first2``, ``last2``: range of elements to compare

  - must be *random access iterators*, e.g., returned from ``Kokkos::Experimental::(c)begin/(c)end``

  - must represent valid ranges, i.e., ``last1 >= first1`` and ``last2 >= first2``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``view1``, ``view2``: views to compare

  - must be rank-1, and have ``LayoutLeft``, ``LayoutRight``, or ``LayoutStride``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``pred``: comparison function object returning ``true`` if the first agument is less than the second

  - must conform to:

  .. code-block:: cpp

     template <class ValueType1, class ValueType2 = ValueType1>
     struct Comp {
      KOKKOS_INLINE_FUNCTION
      bool operator()(const ValueType1& a, const ValueType2& b) const {
        return ...;
      }
     };
