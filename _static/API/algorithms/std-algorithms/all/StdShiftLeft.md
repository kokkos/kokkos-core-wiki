
# `shift_left`

Header File: `Kokkos_StdAlgorithms.hpp`

```c++
namespace Kokkos{
namespace Experimental{

template <class ExecutionSpace, class IteratorType>
IteratorType shift_left(const ExecutionSpace& exespace,                 (1)
                        IteratorType first,
                        IteratorType last,
                        typename IteratorType::difference_type n);

template <class ExecutionSpace, class IteratorType>
IteratorType shift_left(const std::string& label,                       (2)
                        const ExecutionSpace& exespace,
                        IteratorType first, IteratorType last,
                        typename IteratorType::difference_type n);

template <class ExecutionSpace, class DataType, class... Properties>
auto shift_left(const ExecutionSpace& exespace,                         (3)
                const ::Kokkos::View<DataType, Properties...>& view,
                typename decltype(begin(view))::difference_type n);

template <class ExecutionSpace, class DataType, class... Properties>
auto shift_left(const std::string& label,                               (4)
                const ExecutionSpace& exespace,
                const ::Kokkos::View<DataType, Properties...>& view,
                typename decltype(begin(view))::difference_type n);

} //end namespace Experimental
} //end namespace Kokkos
```
## Description

Shifts the elements in the range `[first, last)` or in `view`
by `n` positions towards the *beginning*.

## Parameters and Requirements

- `exespace`:
  - execution space
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::shift_left_iterator_api_default"
  - for 3, the default string is: "Kokkos::shift_left_view_api_default"
- `first, last`:
  - range of elements to shift
  - must be *random access iterators*
  - must represent a valid range, i.e., `last >= first` (checked in debug mode)
  - must be accessible from `exespace`
- `view_from`:
  - view to modify
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`
- `n`:
  - the number of positions to shift
  - must be non-negative

## Return

The end of the resulting range.
If `n` is less than `last - first`, returns `first + (last - first - n)`.
Otherwise, returns `first`.
