
``copy_backward``
=================

Header: `<Kokkos_StdAlgorithms.hpp>`

Description
-----------

Copies the elements in reverse order from range `[first_from, last_from)` to another
range *ending* at `last_to` (overloads 1,2,5) or from
a source view `view_from` to a destination view `view_to` (overloads 3,4,6).
The relative order is preserved.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.

.. code-block:: cpp

  //
  // overload set accepting an execution space
  //
  template <class ExecutionSpace, class InputIteratorType, class OutputIteratorType>
  OutputIteratorType copy_backward(const ExecutionSpace& exespace,                (1)
                                   InputIteratorType first_from,
                                   InputIteratorType last_from,
                                   OutputIteratorType last_to);

  template <class ExecutionSpace, class InputIteratorType, class OutputIteratorType>
  OutputIteratorType copy_backward(const std::string& label,
                                   const ExecutionSpace& exespace,                (2)
                                   InputIteratorType first_from,
                                   InputIteratorType last_from,
                                   OutputIteratorType last_to);

  template <
    class ExecutionSpace,
    class DataType1, class... Properties1,
    class DataType2, class... Properties2
  >
  auto copy_backward(const ExecutionSpace& exespace,                              (3)
                     const Kokkos::View<DataType1, Properties1...>& view_from,
                     const Kokkos::View<DataType2, Properties2...>& view_to);

  template <
    class ExecutionSpace,
    class DataType1, class... Properties1,
    class DataType2, class... Properties2
  >
  auto copy_backward(const std::string& label, const ExecutionSpace& exespace,    (4)
                     const Kokkos::View<DataType1, Properties1...>& view_from,
                     const Kokkos::View<DataType2, Properties2...>& view_to);

  //
  // overload set accepting a team handle
  //
  template <class TeamHandleType, class IteratorType1, class IteratorType2>
  IteratorType2 copy_backward(const TeamHandleType& teamHandle,                   (5)
                              IteratorType1 first,
                              IteratorType1 last, IteratorType2 d_last);

  template <
    class TeamHandleType, class DataType1, class... Properties1,
    class DataType2, class... Properties2>
  auto copy_backward(                                                             (6)
    const TeamHandleType& teamHandle,
    const ::Kokkos::View<DataType1, Properties1...>& source,
    ::Kokkos::View<DataType2, Properties2...>& dest);

Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `exespace`:
  - execution space instance
- `teamHandle`:
  -  team handle instance given inside a parallel region when using a TeamPolicy
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::copy_backward_iterator_api_default"
  - for 3, the default string is: "Kokkos::copy_backward_view_api_default"
  - NOTE: overloads accepting a team handle do not use a label internally
- `first_from, last_from`:
  - range of elements to copy from
  - must be *random access iterators*
  - must represent a valid range, i.e., `last_from >= first_from` (checked in debug mode)
  - must be accessible from `exespace`
- `last_to`:
  - iterator past the last element of the range to copy to
  - must be a *random access iterator*
  - must be accessible from `exespace`
- `view_from`, `view_to`:
  - source and destination views to copy elements from and to
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`


Return Value
~~~~~~~~~~~~

Iterator to the last element copied.
