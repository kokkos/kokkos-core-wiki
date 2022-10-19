
# `adjacent_find`

Header File: `Kokkos_StdAlgorithms.hpp`

```cpp
namespace Kokkos{
namespace Experimental{

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

} //end namespace Experimental
} //end namespace Kokkos
```

## Description

Searches the range `[first,last)` (1,2,5,6) or `view` (3,4,7,8) for two consecutive equal elements.


## Parameters and Requirements

- `exespace`:
  - execution space instance

- `label`:
    - 1,5: The default string is "Kokkos::adjacent_find_iterator_api_default".
    - 3,7: The default string is "Kokkos::adjacent_find_view_api_default".

- `first, last`:
  - range of elements to search in
  - must be *random access iterators*, e.g., returned from `Kokkos::Experimental::(c)begin/(c)end`
  - must represent a valid range, i.e., `last >= first` (this condition is checked in debug mode)
  - must be accessible from `exespace`

- `view`:
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`

- `pred` - similar to [`equal`](./StdEqual)


## Return 

Returns the first iterator `it` in (1,2), where `*it == *it+1` returns true.
Returns the first iterator `it` in (5,6), where `pred(*it, *it+1)` returns true.
Returns the first Kokkos view iterator `it` in (3,4), where `view(it) == view(it+1)` returns true.
Returns the first Kokkos view iterator `it` in (7,8), where `pred(view(it), view(it+1))` returns true.
