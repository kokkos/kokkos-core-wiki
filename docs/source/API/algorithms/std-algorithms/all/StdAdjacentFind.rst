
``adjacent_find``
=================

Header: ``<Kokkos_StdAlgorithms.hpp>``

Description
-----------

Searches a given range or rank-1 ``View`` for two consecutive equal elements.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.


.. code-block:: cpp

   //
   // overload set accepting iterators
   //
   template <class ExecutionSpace, class IteratorType>
   IteratorType adjacent_find(const ExecutionSpace& exespace, IteratorType first,          (1)
			      IteratorType last);

   template <class ExecutionSpace, class IteratorType>
   IteratorType adjacent_find(const std::string& label, const ExecutionSpace& exespace,    (2)
			      IteratorType first, IteratorType last);

   template <class ExecutionSpace, class DataType, class... Properties>
   auto adjacent_find(const ExecutionSpace& exespace,                                      (3)
		      const ::Kokkos::View<DataType, Properties...>& view);

   template <class ExecutionSpace, class DataType, class... Properties>
   auto adjacent_find(const std::string& label, const ExecutionSpace& exespace,            (4)
		      const ::Kokkos::View<DataType, Properties...>& view);

   template <class ExecutionSpace, class IteratorType, class BinaryPredicateType>
   IteratorType adjacent_find(const ExecutionSpace& exespace, IteratorType first,          (5)
			      IteratorType last, BinaryPredicateType pred);

   template <class ExecutionSpace, class IteratorType, class BinaryPredicateType>
   IteratorType adjacent_find(const std::string& label, const ExecutionSpace& exespace,    (6)
			      IteratorType first, IteratorType last,
			      BinaryPredicateType pred);

   template <class ExecutionSpace, class DataType, class... Properties,
	     class BinaryPredicateType>
   auto adjacent_find(const ExecutionSpace& exespace,
		      const ::Kokkos::View<DataType, Properties...>& view,                 (7)
		      BinaryPredicateType pred);

   template <class ExecutionSpace, class DataType, class... Properties,
	     class BinaryPredicateType>
   auto adjacent_find(const std::string& label, const ExecutionSpace& exespace,            (8)
		      const ::Kokkos::View<DataType, Properties...>& view,
		      BinaryPredicateType pred);

   //
   // overload set accepting a team handle
   //
   template <class TeamHandleType, class IteratorType>
   KOKKOS_FUNCTION
   IteratorType adjacent_find(const TeamHandleType& teamHandle, IteratorType first,        (9)
			      IteratorType last);

   template <class TeamHandleType, class DataType, class... Properties>
   KOKKOS_FUNCTION
   auto adjacent_find(const TeamHandleType& teamHandle,                                   (10)
		      const ::Kokkos::View<DataType, Properties...>& view);

   template <class TeamHandleType, class IteratorType, class BinaryPredicateType>
   KOKKOS_FUNCTION
   IteratorType adjacent_find(const TeamHandleType& teamHandle, IteratorType first,       (11)
			      IteratorType last, BinaryPredicateType pred);

   template <class TeamHandleType, class DataType, class... Properties,
	     class BinaryPredicateType>
   KOKKOS_FUNCTION
   auto adjacent_find(const TeamHandleType& teamHandle,
		      const ::Kokkos::View<DataType, Properties...>& view,                (12)
		      BinaryPredicateType pred);


Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``exespace``: execution space instance

- ``teamHandle``: team handle instance given inside a parallel region when using a TeamPolicy

- ``label``:

  - 1,5: The default string is "Kokkos::adjacent_find_iterator_api_default".

  - 3,7: The default string is "Kokkos::adjacent_find_view_api_default".

- ``first, last``: range of elements to search in

  - must be *random access iterators*, e.g., returned from ``Kokkos::Experimental::(c)begin/(c)end``

  - must represent a valid range, i.e., ``last >= first`` (asserted in debug mode)

  - must be accessible from ``exespace`` or from the execution space associated with the team handle
    (this check happens at compile-time)

- ``view``:

  - must be rank-1, and have ``LayoutLeft``, ``LayoutRight``, or ``LayoutStride``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle
    (this check happens at compile-time)

- ``pred``: *binary* functor returning ``true`` if two arguments should be considered "equal".
  ``pred(a,b)`` must be valid to be called from the execution space passed, or
  the execution space associated with the team handle, and convertible to bool
  for every pair of arguments ``a,b`` of type ``value_type``, where ``value_type``
  is the value type of ``IteratorType`` or ``view`` and must not modify ``a,b``.

  - must conform to:

  .. code-block:: cpp

     struct Comparator
     {
       KOKKOS_INLINE_FUNCTION
       bool operator()(const value_type & a, const value_type & b) const {
         return /* true if a should be considered equal to b */;
       }
     };


Return Value
~~~~~~~~~~~~

- 1,2,9: returns the first iterator ``it`` where ``*it == *it+1`` is true
- 5,6,11: returns the first iterator ``it`` where ``pred(*it, *it+1)`` is true
- 3,4,10: returns the first Kokkos view iterator ``it`` where ``view(it) == view(it+1)`` is true
- 7,8,12: returns the first Kokkos view iterator ``it`` ), where ``pred(view(it), view(it+1))`` is true

If no such element is found, it returns ``last`` for all overloads accepting iterators,
and ``Kokkos::Experimental::end(view)`` for all overloads acceptings a view.
