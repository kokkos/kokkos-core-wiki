``count``
=========

Header: ``<Kokkos_StdAlgorithms.hpp>``

Description
-----------

Returns the number of elements in a range or in rank-1 ``View`` that are equal to a target value.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.

.. code-block:: cpp

   //
   // overload set accepting iterators
   //
   template <class ExecutionSpace, class IteratorType, class T>
   typename IteratorType::difference_type count(const ExecutionSpace& exespace,
						IteratorType first,
						IteratorType last,                      (1)
						const T& value);

   template <class ExecutionSpace, class IteratorType, class T>
   typename IteratorType::difference_type count(const std::string& label,
						const ExecutionSpace& exespace,
						IteratorType first,
						IteratorType last,                      (2)
						const T& value);

   template <class ExecutionSpace, class DataType, class... Properties, class T>
   auto count(const ExecutionSpace& exespace,                                           (3)
	      const ::Kokkos::View<DataType, Properties...>& view, const T& value);

   template <class ExecutionSpace, class DataType, class... Properties, class T>
   auto count(const std::string& label, const ExecutionSpace& exespace,                 (4)
	      const ::Kokkos::View<DataType, Properties...>& view,
	      const T& value);

   //
   // overload set accepting a team handle
   //
   template <class TeamHandleType, class IteratorType, class T>
   KOKKOS_FUNCTION
   typename IteratorType::difference_type count(const TeamHandleType& teamHandle,
						IteratorType first,
						IteratorType last,                      (5)
						const T& value);

   template <class TeamHandleType, class DataType, class... Properties, class T>
   KOKKOS_FUNCTION
   auto count(const TeamHandleType& teamHandle,                                         (6)
	      const ::Kokkos::View<DataType, Properties...>& view,
	      const T& value);


Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``exespace``: execution space instance

- ``teamHandle``: team handle instance given inside a parallel region when using a TeamPolicy

- ``label``:

  - 1: The default string is "Kokkos::count_iterator_api_default".

  - 3: The default string is "Kokkos::count_view_api_default".

- ``first, last``: range of elements to search in

  - must be *random access iterators*, e.g., returned from ``Kokkos::Experimental::(c)begin/(c)end``

  - must represent a valid range, i.e., ``last >= first`` (asserted in debug mode)

  - must be accessible from ``exespace`` or from the execution space associated with the team handle
    (this check happens at compile-time)

- ``view``:

  - must be rank-1, and have ``LayoutLeft``, ``LayoutRight``, or ``LayoutStride``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

Return Value
~~~~~~~~~~~~

Returns the number of elements in the range ``first,last`` or in ``view`` that are equal to ``value``.
