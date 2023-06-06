
``copy_if``
===========

Header: `<Kokkos_StdAlgorithms.hpp>`

Description
-----------

Copies the elements for which `pred` returns `true` from range `[first_from, last_from)`
to another range beginning at `first_to` (overloads 1,2,5) or from `view_from` to `view_to`
(overloads 3,4,6).

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.

.. code-block:: cpp

  //
  // overload set accepting an execution space
  //
  template <
    class ExecutionSpace, class InputIteratorType, class OutputIteratorType, class UnaryPredicateType
  >
  OutputIteratorType copy_if(const ExecutionSpace& exespace,                   (1)
                             InputIteratorType first_from,
                             InputIteratorType last_from,
                             OutputIteratorType first_to,
                             UnaryPredicateType pred);

  template <
    class ExecutionSpace, class InputIteratorType, class OutputIteratorType, class UnaryPredicateType
  >
  OutputIteratorType copy_if(const std::string& label,
                             const ExecutionSpace& exespace,                   (2)
                             InputIteratorType first_from,
                             InputIteratorType last_from,
                             OutputIteratorType first_to,
                             UnaryPredicateType pred);

  template <
    class ExecutionSpace,
    class DataType1, class... Properties1,
    class DataType2, class... Properties2,
    class UnaryPredicateType
  >
  auto copy_if(const ExecutionSpace& exespace,                                 (3)
               const Kokkos::View<DataType1, Properties1...>& view_from,
               const Kokkos::View<DataType2, Properties2...>& view_to,
               UnaryPredicateType pred);

  template <
    class ExecutionSpace,
    class DataType1, class... Properties1,
    class DataType2, class... Properties2,
    class UnaryPredicateType
  >
  auto copy_if(const std::string& label, const ExecutionSpace& exespace,       (4)
               const Kokkos::View<DataType1, Properties1...>& view_from,
               const Kokkos::View<DataType2, Properties2...>& view_to,
               UnaryPredicateType pred);

  //
  // overload set accepting a team handle
  //
  template <class TeamHandleType, class InputIterator, class Size,
          class OutputIterator>
  OutputIterator copy_n(const TeamHandleType& teamHandle, InputIterator first, (5)
                        Size count, OutputIterator result);

  template <
    class TeamHandleType, class DataType1, class... Properties1, class Size,
    class DataType2, class... Properties2>
  auto copy_n(                                                                 (6)
    const TeamHandleType& teamHandle,
    const ::Kokkos::View<DataType1, Properties1...>& source, Size count,
    ::Kokkos::View<DataType2, Properties2...>& dest);


Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `exespace`, `first_from`, `last_from`, `first_to`, `view_from`, `view_to`:
  - same as in [`copy`](./StdCopy)
- `teamHandle`:
  -  team handle instance given inside a parallel region when using a TeamPolicy
  - NOTE: overloads accepting a team handle do not use a label internally
- `label`:
  - for 1, the default string is: "Kokkos::copy_if_iterator_api_default"
  - for 3, the default string is: "Kokkos::copy_if_view_api_default"
- `pred`:
  - unary predicate which returns `true` for the required element; `pred(v)`
  must be valid to be called from the execution space passed, and convertible to bool for every
  argument `v` of type (possible const) `value_type`, where `value_type`
  is the value type of `InputIteratorType` (for 1,2) or of `view_from` (for 3,4),
  and must not modify `v`.
  - should have the same API as the unary predicate in [`replace_if`](./StdReplaceIf)


Return Value
~~~~~~~~~~~~

Iterator to the destination element *after* the last element copied.
