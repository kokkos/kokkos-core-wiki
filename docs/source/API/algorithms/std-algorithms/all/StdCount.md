
# `count`

Header File: `Kokkos_StdAlgorithms.hpp`

```cpp
namespace Kokkos{
namespace Experimental{

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
           const ::Kokkos::View<DataType, Properties...>& view, const T& value);

} //end namespace Experimental
} //end namespace Kokkos
```

## Description

Returns the number of elements in the range `[first,last)` (1,2) or in `view` (3,4) that are equal to `value`.

## Parameters and Requirements

- `exespace`:
  - execution space instance

- `label`:
    - 1: The default string is "Kokkos::count_iterator_api_default".
    - 3: The default string is "Kokkos::count_view_api_default".

- `first, last`:
  - range of elements to search in
  - must be *random access iterators*, e.g., returned from `Kokkos::Experimental::(c)begin/(c)end`
  - must represent a valid range, i.e., `last >= first` (this condition is checked in debug mode)
  - must be accessible from `exespace`

- `view`:
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`