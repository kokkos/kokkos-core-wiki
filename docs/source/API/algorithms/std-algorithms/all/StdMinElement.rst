
``min_element``
===============

Header File: ``Kokkos_StdAlgorithms.hpp``


.. code-block:: cpp

   namespace Kokkos{
   namespace Experimental{

   //
   // overload set accepting iterators
   //
   template <class ExecutionSpace, class IteratorType>
   auto min_element(const ExecutionSpace& exespace,                          (1)
		    IteratorType first, IteratorType last);

   template <class ExecutionSpace, class IteratorType>
   auto min_element(const std::string& label,                                (2)
		    const ExecutionSpace& exespace,
		    IteratorType first, IteratorType last);

   template <class ExecutionSpace, class IteratorType, class ComparatorType>
   auto min_element(const ExecutionSpace& exespace,                          (3)
		    IteratorType first, IteratorType last,
		    ComparatorType comp);

   template <class ExecutionSpace, class IteratorType, class ComparatorType>
   auto min_element(const std::string& label,                                (4)
		    const ExecutionSpace& exespace,
		    IteratorType first, IteratorType last,
		    ComparatorType comp);

   template <class TeamHandleType, class IteratorType>
   KOKKOS_FUNCTION
   auto min_element(const TeamHandleType& teamHandle,                        (5)
		    IteratorType first, IteratorType last);

   template <class TeamHandleType, class IteratorType, class ComparatorType>
   KOKKOS_FUNCTION
   auto min_element(const TeamHandleType& teamHandle,                        (6)
		    IteratorType first, IteratorType last,
		    ComparatorType comp);

   //
   // overload set accepting views
   //
   template <class ExecutionSpace, class DataType, class... Properties>
   auto min_element(const ExecutionSpace& exespace,                          (7)
		    const ::Kokkos::View<DataType, Properties...>& view);

   template <class ExecutionSpace, class DataType, class... Properties>
   auto min_element(const std::string& label,                                (8)
		    const ExecutionSpace& exespace,
		    const ::Kokkos::View<DataType, Properties...>& view);

   template <class ExecutionSpace, class DataType, class ComparatorType, class... Properties>
   auto min_element(const ExecutionSpace& exespace,                          (9)
		    const ::Kokkos::View<DataType, Properties...>& view,
		    ComparatorType comp);

   template <class ExecutionSpace, class DataType, class ComparatorType, class... Properties>
   auto min_element(const std::string& label,                               (10)
		    const ExecutionSpace& exespace,
		    const ::Kokkos::View<DataType, Properties...>& view,
		    ComparatorType comp);

   template <class TeamHandleType, class DataType, class... Properties>
   KOKKOS_FUNCTION
   auto min_element(const TeamHandleType& teamHandle,                       (11)
		    const ::Kokkos::View<DataType, Properties...>& view);

   template <class TeamHandleType, class DataType, class ComparatorType, class... Properties>
   KOKKOS_FUNCTION
   auto min_element(const TeamHandleType& teamHandle,                       (12)
		    const ::Kokkos::View<DataType, Properties...>& view,
		    ComparatorType comp);

   } //end namespace Experimental
   } //end namespace Kokkos


Description
-----------

- (1,2,5,7,8,11): Finds the smallest element in the range ``[first, last)`` (1,2,5)
  or in ``view`` (7,8,11) using ``operator<`` to compare two elements

- (3,4,6,9,10,12): Finds the smallest element in the range ``[first, last)`` (3,4,6)
  or in ``view`` (7,8,12) using the binary functor ``comp`` to compare two elements


Parameters and Requirements
---------------------------

- ``exespace``:
  - execution space instance

- ``teamHandle``:
  - team handle as given inside a parallel region executed through a TeamPolicy

- ``label``:
  - used to name the implementation kernels for debugging purposes

  - for 1,3 the default string is: "Kokkos::min_element_iterator_api_default"

  - for 5,7 the default string is: "Kokkos::min_element_view_api_default"

- ``first``, ``last``:

  - range of elements to examine

  - must be *random access iterators*

  - must represent a valid range, i.e., ``last >= first`` (checked in debug mode)

  - must be accessible from ``exespace`` (for 1-4, 7-10),
    or from the execution space associated with the team handle (5,6,11,12)

- ``view``:

  - Kokkos view to examine

  - must be rank-1, and have ``LayoutLeft``, ``LayoutRight``, or ``LayoutStride``

  - must be accessible from ``exespace`` ,
    or from the execution space associated with the team handle (5,6,11,12)


- ``comp``:

  - *binary* functor returning ``true`` if the first argument is *less than* the second argument;
    ``comp(a,b)`` must be valid to be called from the execution space passed,
    and convertible to bool for every pair of arguments ``a,b`` of type
    ``value_type``, where ``value_type`` is the value type of ``IteratorType`` (for 1-6)
    or the value type of ``view`` (for 7-12) and must not modify ``a,b``.

  - must conform to:

    .. code-block:: cpp

       struct Comparator
       {
	  KOKKOS_INLINE_FUNCTION
	  bool operator()(const value_type & a, const value_type & b) const {
		return /* true if a is less than b, based on your logic of "less than" */;
	  }
       };

Return
------

Iterator to the smallest element.
The following special cases apply:

- if several elements are equivalent to the smallest element, it returns the iterator to the *first* such element.

- if the range ``[first, last)`` is empty it returns ``last``.

- if ``view`` is empty, it returns ``Kokkos::Experimental::end(view)``.


Example: execution space API
----------------------------

.. code-block:: cpp

   namespace KE = Kokkos::Experimental;
   Kokkos::View<double*> a("a", 13);
   // fill a somehow

   auto res = KE::min_element(Kokkos::DefaultExecutionSpace(), KE::begin(a), KE::end(a));

   // passing the view directly
   auto res = KE::min_element(Kokkos::DefaultExecutionSpace(), a);

   // using a custom comparator
   template <class ValueType1, class ValueType2 = ValueType1>
   struct CustomLessThanComparator {
     KOKKOS_INLINE_FUNCTION
     bool operator()(const ValueType1& a,
		     const ValueType2& b) const {
      // here we use < but one can put any custom logic to return true if a is less than b
       return a < b;
     }

     KOKKOS_INLINE_FUNCTION
     CustomLessThanComparator() {}
   };

   // passing the view directly
   auto res = KE::min_element(Kokkos::DefaultExecutionSpace(), a, CustomLessThanComparator<double>());



..
   when available we should use this

   Example: team level API
   -----------------------

   .. literalinclude:: https://github.com/fnrizzi/kokkos/blob/std_replace_team_impl/algorithms/unit_tests/TestStdAlgorithmsTeamMinElement.cpp
      :language: cpp
      :lines: 52-162
