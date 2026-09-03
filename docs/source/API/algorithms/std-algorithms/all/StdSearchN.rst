
``search_n``
============

Header: ``<Kokkos_StdAlgorithms.hpp>``

Description
-----------

Searches a range or a ``View`` for the first sequence of ``count`` identical elements each equal to the given value.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.

Overload set accepting execution space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <class ExecutionSpace, class IteratorType, class SizeType,
	     class ValueType>
   IteratorType search_n(const ExecutionSpace& exespace, IteratorType first,
			 IteratorType last, SizeType count,                             (1)
			 const ValueType& value);

   template <class ExecutionSpace, class IteratorType, class SizeType,
	     class ValueType>
   IteratorType search_n(const std::string& label, const ExecutionSpace& exespace,
			 IteratorType first, IteratorType last, SizeType count,         (2)
			 const ValueType& value);

   template <class ExecutionSpace, class DataType, class... Properties,
	     class SizeType, class ValueType>
   auto search_n(const ExecutionSpace& exespace,
		 const ::Kokkos::View<DataType, Properties...>& view,                   (3)
		 SizeType count, const ValueType& value);

   template <class ExecutionSpace, class DataType, class... Properties,
	     class SizeType, class ValueType>
   auto search_n(const std::string& label, const ExecutionSpace& exespace,
		 const ::Kokkos::View<DataType, Properties...>& view,                   (4)
		 SizeType count, const ValueType& value);

   // overload set 2: binary predicate passed
   template <class ExecutionSpace, class IteratorType, class SizeType,
	     class ValueType, class BinaryPredicateType>
   IteratorType search_n(const ExecutionSpace& exespace, IteratorType first,
			 IteratorType last, SizeType count, const ValueType& value,     (5)
			 const BinaryPredicateType& pred);

   template <class ExecutionSpace, class IteratorType, class SizeType,
	     class ValueType, class BinaryPredicateType>
   IteratorType search_n(const std::string& label, const ExecutionSpace& exespace,
			 IteratorType first, IteratorType last, SizeType count,         (6)
			 const ValueType& value, const BinaryPredicateType& pred);

   template <class ExecutionSpace, class DataType, class... Properties,
	     class SizeType, class ValueType, class BinaryPredicateType>
   auto search_n(const ExecutionSpace& exespace,
		 const ::Kokkos::View<DataType, Properties...>& view,                   (7)
		 SizeType count, const ValueType& value,
		 const BinaryPredicateType& pred);

   template <class ExecutionSpace, class DataType, class... Properties,
	     class SizeType, class ValueType, class BinaryPredicateType>
   auto search_n(const std::string& label, const ExecutionSpace& exespace,
		 const ::Kokkos::View<DataType, Properties...>& view,                   (8)
		 SizeType count, const ValueType& value,
		 const BinaryPredicateType& pred);

Overload set accepting a team handle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 4.2

.. code-block:: cpp

   template <class TeamHandleType, class IteratorType, class SizeType,
	     class ValueType>
   KOKKOS_FUNCTION
   IteratorType search_n(const TeamHandleType& teamHandle, IteratorType first,
			 IteratorType last, SizeType count,                             (9)
			 const ValueType& value);

   template <class TeamHandleType, class DataType, class... Properties,
	     class SizeType, class ValueType>
   KOKKOS_FUNCTION
   auto search_n(const TeamHandleType& teamHandle,
		 const ::Kokkos::View<DataType, Properties...>& view,                  (10)
		 SizeType count, const ValueType& value);

   // overload set 2: binary predicate passed
   template <class TeamHandleType, class IteratorType, class SizeType,
	     class ValueType, class BinaryPredicateType>
   KOKKOS_FUNCTION
   IteratorType search_n(const TeamHandleType& teamHandle, IteratorType first,
			 IteratorType last, SizeType count, const ValueType& value,    (11)
			 const BinaryPredicateType& pred);

   template <class TeamHandleType, class DataType, class... Properties,
	     class SizeType, class ValueType, class BinaryPredicateType>
   KOKKOS_FUNCTION
   auto search_n(const TeamHandleType& teamHandle,
		 const ::Kokkos::View<DataType, Properties...>& view,                  (12)
		 SizeType count, const ValueType& value,
		 const BinaryPredicateType& pred);

Detailed Description
~~~~~~~~~~~~~~~~~~~~

- Searches the range ``[first, last)`` for a range of ``count`` elements
  each comparing equal to ``value``  (1,2,9).

- Searches the ``view`` for ``count`` elements each comparing equal to ``value``  (3,4,10).

- Searches the range [first, last) for a range of ``count`` elements
  for which the ``pred`` returns true for ``value`` in (5,6,11).

- Searches the ``view`` for a range of ``count`` elements for which
  the ``pred`` returns true for ``value`` in (7,8,12).

Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``exespace``: execution space instance

- ``teamHandle``: team handle instance given inside a parallel region when using a TeamPolicy

- ``label``: string forwarded to internal parallel kernels for debugging purposes

  - 1,5: The default string is "Kokkos::search_n_iterator_api_default".

  - 3,7: The default string is "Kokkos::search_n_view_api_default".

  - NOTE: overloads accepting a team handle do not use a label internally

- ``n``: number of elements to operate on

- ``first``: iterator defining the beginning of range

  - must be *random access iterator*

  - ``[first, first+count)`` must represent a valid range

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``view``:

  - must be rank-1, and have ``LayoutLeft``, ``LayoutRight``, or ``LayoutStride``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``pred``: *binary* functor returning ``true`` if two arguments should be considered "equal".

  ``pred(a,b)`` must be valid to be called from the execution space passed, or
  the execution space associated with the team handle, and convertible to bool
  for every pair of arguments ``a,b`` of type ``ValueType1`` and ``ValueType``,
  respectively, where ``ValueType1`` is the value type of ``IteratorType`` or ``view``,
  and must not modify ``a,b``.

  - must conform to:

  .. code-block:: cpp

     template <class ValueType1, class ValueType2 = ValueType1>
     struct IsEqualFunctor {
      KOKKOS_INLINE_FUNCTION
      bool operator()(const ValueType1& a, const ValueType2& b) const {
        return (a == b);
      }
     };
