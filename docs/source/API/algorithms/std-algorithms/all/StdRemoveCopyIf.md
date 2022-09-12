
# `remove_copy_if`

Header File: `Kokkos_StdAlgorithms.hpp`

```c++
namespace Kokkos{
namespace Experimental{

template <
  class ExecutionSpace,
  class InputIterator, class OutputIterator,
  class UnaryPredicateType>
OutputIterator remove_copy_if(const ExecutionSpace& exespace,                   (1)
                              InputIterator first_from,
                              InputIterator last_from,
                              OutputIterator first_to,
                              UnaryPredicateType pred);

template <
  class ExecutionSpace,
  class InputIterator, class OutputIterator,
  class UnaryPredicateType>
OutputIterator remove_copy_if(const std::string& label,                         (2)
                           const ExecutionSpace& exespace,
                           InputIterator first_from,
                           InputIterator last_from,
                           OutputIterator first_to,
                           UnaryPredicateType pred);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class UnaryPredicateType>
auto remove_copy_if(const ExecutionSpace& exespace,                             (3)
                    const ::Kokkos::View<DataType1, Properties1...>& view_from,
                    const ::Kokkos::View<DataType2, Properties2...>& view_dest,
                    UnaryPredicateType pred);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class UnaryPredicateType>
auto remove_copy_if(const std::string& label,                                   (4)
                    const ExecutionSpace& exespace,
                    const ::Kokkos::View<DataType1, Properties1...>& view_from,
                    const ::Kokkos::View<DataType2, Properties2...>& view_dest,
                    UnaryPredicateType pred);

} //end namespace Experimental
} //end namespace Kokkos
```

## Description

Copies the elements from the range `[first_from, last_from)` to a new
range starting at `first_to` or from `view_from` to `view_dest` omitting
those for which `pred` returns `true`.


## Parameters and Requirements

- `exespace`, `first_from, last_from`, `first_to`, `view_from`, `view_dest`: same as in [`remove_copy`](./StdRemoveCopy)
- `label`:
  - for 1, the default string is: "Kokkos::remove_copy_if_iterator_api_default"
  - for 3, the default string is: "Kokkos::remove_copy_if_view_api_default"
- `pred`:
  - *unary* predicate which returns `true` for the required element; `pred(v)`
  must be valid to be called from the execution space passed, and convertible to bool for every
  argument `v` of type (possible const) `value_type`, where `value_type`
  is the value type of `InputIteratorType` (for 1,2) or of `view_from` (for 3,4),
  and must not modify `v`.
  - must conform to:
  ```c++
  struct Predicate
  {
     KOKKOS_INLINE_FUNCTION
     bool operator()(const value_type & v) const { return /* ... */; }

     // or, also valid

     KOKKOS_INLINE_FUNCTION
     bool operator()(value_type v) const { return /* ... */; }
  };
  ```


## Return

Iterator to the element after the last element copied.
