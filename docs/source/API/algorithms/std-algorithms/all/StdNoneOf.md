
# `none_of`

Header File: `Kokkos_StdAlgorithms.hpp`

```cpp
namespace Kokkos{
namespace Experimental{

template <class ExecutionSpace, class IteratorType, class Predicate>
bool none_of(const ExecutionSpace& exespace, IteratorType first, IteratorType last,  (1)
             Predicate predicate);

template <class ExecutionSpace, class IteratorType, class Predicate>
bool none_of(const std::string& label, const ExecutionSpace& exespace,               (2)
             IteratorType first, IteratorType last, Predicate predicate);

template <class ExecutionSpace, class DataType, class... Properties,
          class Predicate>
bool none_of(const ExecutionSpace& exespace,
             const ::Kokkos::View<DataType, Properties...>& v,                       (3)
             Predicate predicate);

template <class ExecutionSpace, class DataType, class... Properties,
          class Predicate>
bool none_of(const std::string& label, const ExecutionSpace& exespace,               (4)
             const ::Kokkos::View<DataType, Properties...>& v,
             Predicate predicate);

} //end namespace Experimental
} //end namespace Kokkos
```
## Description

Returns `true` if no element in the range `[first,last)` (1,2) or in `view` (3,4) 
satisfies the unary predicate. 

## Parameters and Requirements

- `exespace`:
  - execution space instance

- `label`:
    - 1: The default string is "Kokkos::none_of_iterator_api_default".
    - 3: The default string is "Kokkos::none_of_view_api_default".

- `first, last`:
  - range of elements to check
  - must be *random access iterators*, e.g., returned from `Kokkos::Experimental::(c)begin/(c)end`
  - must represent a valid range, i.e., `last >= first` (this condition is checked in debug mode)
  - must be accessible from `exespace`

- `view`:
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`

- `pred` - similar to [`count`](./StdCount)
