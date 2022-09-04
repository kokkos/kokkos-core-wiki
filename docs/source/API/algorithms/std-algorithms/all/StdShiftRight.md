
# `shift_right`

Header File: `Kokkos_StdAlgorithms.hpp`

```c++
namespace Kokkos{
namespace Experimental{

template <class ExecutionSpace, class IteratorType>
IteratorType shift_right(const ExecutionSpace& exespace,                  (1)
                         IteratorType first,
                         IteratorType last,
                         typename IteratorType::difference_type n);

template <class ExecutionSpace, class IteratorType>
IteratorType shift_right(const std::string& label,                        (2)
                         const ExecutionSpace& exespace,
                         IteratorType first, IteratorType last,
                         typename IteratorType::difference_type n);

template <class ExecutionSpace, class DataType, class... Properties>
auto shift_right(const ExecutionSpace& exespace,                          (3)
                 const ::Kokkos::View<DataType, Properties...>& view,
                 typename decltype(begin(view))::difference_type n);

template <class ExecutionSpace, class DataType, class... Properties>
auto shift_right(const std::string& label,                                (4)
                 const ExecutionSpace& exespace,
                 const ::Kokkos::View<DataType, Properties...>& view,
                 typename decltype(begin(view))::difference_type n);

} //end namespace Experimental
} //end namespace Kokkos
```
## Description

Shifts the elements in the range `[first, last)` or in `view`
by `n` positions towards the end of the range or the view.

## Parameters and Requirements

- `exespace`, `first`, `last`, `view`: same as in [`shift_left`](./StdShiftLeft)
- `label`:
  - for 1, the default string is: "Kokkos::shift_right_iterator_api_default"
  - for 3, the default string is: "Kokkos::shift_right_view_api_default"
- `n`:
  - the number of positions to shift
  - must be non-negative

## Return

The beginning of the resulting range. If `n` is less than `last - first`,
returns `first + n`. Otherwise, returns `last`.
