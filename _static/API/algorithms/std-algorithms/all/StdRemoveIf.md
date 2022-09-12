
# `remove_if`

Header File: `Kokkos_StdAlgorithms.hpp`

```c++
namespace Kokkos{
namespace Experimental{

template <class ExecutionSpace, class Iterator, class UnaryPredicateType>
Iterator remove_if(const ExecutionSpace& exespace,                           (1)
                   Iterator first, Iterator last,
                   UnaryPredicateType pred);

template <class ExecutionSpace, class Iterator, class UnaryPredicateType>
Iterator remove_if(const std::string& label,                                 (2)
                   const ExecutionSpace& exespace,
                   Iterator first, Iterator last,
                   UnaryPredicateType pred);

template <
  class ExecutionSpace,
  class DataType, class... Properties,
  class UnaryPredicateType>
auto remove_if(const ExecutionSpace& exespace,                               (3)
               const ::Kokkos::View<DataType, Properties...>& view,
               const UnaryPredicateType& pred);

template <
  class ExecutionSpace,
  class DataType, class... Properties,
  class UnaryPredicateType>
auto remove_if(const std::string& label,                                     (4)
         const ExecutionSpace& exespace,
         const ::Kokkos::View<DataType, Properties...>& view,
         const UnaryPredicateType& pred);

} //end namespace Experimental
} //end namespace Kokkos
```

## Description

Removes all elements for which `pred` returns `true`, by shifting via move assignment
the elements in the range `[first, last)` (1,2) or in `view` (3,4)
such that the elements not to be removed
appear in the beginning of the range (1,2) or in the beginning of `view` (3,4).
Relative order of the elements that remain is preserved
and the physical size of the container is unchanged.


## Parameters and Requirements

- `exespace`, `first`, `last`, `view`: same as in [`remove`](./StdRemove)
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::remove_if_iterator_api_default"
  - for 3, the default string is: "Kokkos::remove_if_view_api_default"
- `pred`:
  - *unary* predicate which returns `true` for the required element; `pred(v)`
  must be valid to be called from the execution space passed, and convertible to bool for every
  argument `v` of type (possible const) `value_type`, where `value_type`
  is the value type of `IteratorType` (for 1,2) or of `view` (for 3,4),
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

Iterator to the element *after* the new logical end.
