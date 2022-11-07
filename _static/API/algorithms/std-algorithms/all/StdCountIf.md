
# `count_if`

Header File: `Kokkos_StdAlgorithms.hpp`

```cpp
namespace Kokkos{
namespace Experimental{

template <class ExecutionSpace, class IteratorType, class Predicate>
typename IteratorType::difference_type count_if(const ExecutionSpace& exespace,
                                                IteratorType first,
                                                IteratorType last,                   (1)
                                                Predicate pred);


template <class ExecutionSpace, class IteratorType, class Predicate>
typename IteratorType::difference_type count_if(const std::string& label,
                                                const ExecutionSpace& exespace,
                                                IteratorType first,                  (2)
                                                IteratorType last,
                                                Predicate pred);

template <class ExecutionSpace, class DataType, class... Properties,
          class Predicate>
auto count_if(const ExecutionSpace& exespace,
              const ::Kokkos::View<DataType, Properties...>& view,                   (3)
              Predicate pred);

template <class ExecutionSpace, class DataType, class... Properties,
          class Predicate>
auto count_if(const std::string& label, const ExecutionSpace& exespace,
              const ::Kokkos::View<DataType, Properties...>& view,                   (4)
              Predicate pred);

} //end namespace Experimental
} //end namespace Kokkos
```

## Description

Returns the number of elements in the range `[first,last)` (1,2) or in `view` (3,4) 
for which `pred` is true.

## Parameters and Requirements

- `exespace`:
  - execution space instance

- `label`:
  - 1: The default string is "Kokkos::count_if_iterator_api_default"
  - 3: The default string is "Kokkos::count_if_view_api_default"

- `first, last`:
  - range of elements to search in
  - must be *random access iterators*, e.g., returned from `Kokkos::Experimental::(c)begin/(c)end`
  - must represent a valid range, i.e., `last >= first` (this condition is checked in debug mode)
  - must be accessible from `exespace`

- `view`:
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`

- `pred` is similar to [`equal`](./StdEqual)

