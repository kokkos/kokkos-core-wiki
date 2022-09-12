
# `move_backward`

Header File: `Kokkos_StdAlgorithms.hpp`

```c++
namespace Kokkos{
namespace Experimental{

template <class ExecutionSpace, class InputIterator, class OutputIterator>
OutputIterator move_backward(const ExecutionSpace& exespace,                 (1)
                             InputIterator first_from,
                             InputIterator last_from,
                             OutputIterator first_to);

template <class ExecutionSpace, class InputIterator, class OutputIterator>
OutputIterator move_backward(const std::string& label,                       (2)
                             const ExecutionSpace& exespace,
                             InputIterator first_from,
                             InputIterator last_from,
                             OutputIterator first_to);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2
>
auto move_backward(const ExecutionSpace& exespace,                           (3)
                   const Kokkos::View<DataType1, Properties1...>& source,
                   Kokkos::View<DataType2, Properties2...>& dest);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2
>
auto move_backward(const std::string& label,                                 (4)
                   const ExecutionSpace& exespace,
                   const Kokkos::View<DataType1, Properties1...>& source,
                   Kokkos::View<DataType2, Properties2...>& dest);

} //end namespace Experimental
} //end namespace Kokkos
```

## Description

- Overloads 1,2: moves the elements from the range `[first_from, last_from)`
in reverse order to the range beginning at `first_to`

- Overloads 3,4: moves the elements from `source` view in reverse order to `dest` view


## Parameters and Requirements

- `exespace`, `first_from`, `last_from`, `source`, `dest`: same as `Kokkos::Experimental::move`
- `label`:
  - for 1, the default string is: "Kokkos::move_backward_iterator_api_default"
  - for 3, the default string is: "Kokkos::move_backward_view_api_default"

## Return

Iterator to the element *after* the last element moved.
