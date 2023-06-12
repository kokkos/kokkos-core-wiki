``minmax_element``
==================

Header: ``Kokkos_StdAlgorithms.hpp``

Description
-----------

Finds the smallest and largest elements in a range or in a rank-1 ``View`` using either ``operator<`` to compare two elements or a user-provided comparison operator.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.

.. code-block:: cpp

   //
   // overload set accepting iterators
   //
   template <class ExecutionSpace, class IteratorType>
   auto minmax_element(const ExecutionSpace& exespace,                        (1)
                       IteratorType first, IteratorType last);

   template <class ExecutionSpace, class IteratorType>
   auto minmax_element(const std::string& label,                              (2)
                       const ExecutionSpace& exespace,
                       IteratorType first, IteratorType last);

   template <class ExecutionSpace, class IteratorType, class ComparatorType>
   auto minmax_element(const ExecutionSpace& exespace,                        (3)
                       IteratorType first, IteratorType last,
                       ComparatorType comp);

   template <class ExecutionSpace, class IteratorType, class ComparatorType>
   auto minmax_element(const std::string& label,                              (4)
                       const ExecutionSpace& exespace,
                       IteratorType first, IteratorType last,
                       ComparatorType comp);

   //
   // overload set accepting views
   //
   template <class ExecutionSpace, class DataType, class... Properties>
   auto minmax_element(const ExecutionSpace& exespace,                        (5)
                       const ::Kokkos::View<DataType, Properties...>& view);

   template <class ExecutionSpace, class DataType, class... Properties>
   auto minmax_element(const std::string& label,                              (6)
                       const ExecutionSpace& exespace,
                       const ::Kokkos::View<DataType, Properties...>& view);

   template <class ExecutionSpace, class DataType, class ComparatorType, class... Properties>
   auto minmax_element(const ExecutionSpace& exespace,                        (7)
                       const ::Kokkos::View<DataType, Properties...>& view,
                       ComparatorType comp);

   template <class ExecutionSpace, class DataType, class ComparatorType, class... Properties>
   auto minmax_element(const std::string& label,                              (8)
                       const ExecutionSpace& exespace,
                       const ::Kokkos::View<DataType, Properties...>& view,
                       ComparatorType comp);

Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _min_element_link: ./StdMinElement.html

.. |min_element_link| replace:: ``min_element``

- ``exespace``, ``first``, ``last``, ``view``, ``comp``: same as in |min_element_link|_

- ``label``: string forwarded to internal parallel kernels for debugging purposes

  - 1 and 3: The default string is "Kokkos::minmax_element_iterator_api_default".

  - 5 and 7: The default string is "Kokkos::minmax_element_view_api_default".

Return Value
~~~~~~~~~~~~

A Kokkos pair of iterators to the smallest and largest elements in that order.

The following special cases apply:

- if the range ``[first, last)`` is empty it returns ``Kokkos::pair(first, first)``.

- if ``view`` is empty, it returns ``Kokkos::pair(Kokkos::Experimental::begin(view), Kokkos::Experimental::begin(view))``.

- if several elements are equivalent to the smallest element, the iterator to the *first* such element is returned.

- if several elements are equivalent to the largest element, the iterator to the *last* such element is returned.

Example
~~~~~~~

.. code-block:: cpp

   namespace KE = Kokkos::Experimental;
   Kokkos::View<double*> a("a", 11);

   auto itPair = KE::minmax_element(Kokkos::DefaultExecutionSpace(), KE::begin(a), KE::end(a));

   // passing the view directly
   auto itPair = KE::minmax_element(Kokkos::DefaultExecutionSpace(), a);


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
   auto res = KE::minmax_element(Kokkos::DefaultExecutionSpace(), a, CustomLessThanComparator<double>());
