
# `rotate_copy`

Header File: `Kokkos_StdAlgorithms.hpp`

```c++
namespace Kokkos{
namespace Experimental{

template <class ExecutionSpace, class InputIterator, class OutputIterator>
OutputIterator rotate_copy(const ExecutionSpace& exespace,                   (1)
                           InputIterator first_from,
                           InputIterator n_first,
                           InputIterator last_from,
                           OutputIterator first_to);

template <class ExecutionSpace, class InputIterator, class OutputIterator>
OutputIterator rotate_copy(const std::string& label,                         (2)
                           const ExecutionSpace& exespace,
                           InputIterator first_from,
                           InputIterator n_first,
                           InputIterator last_from,
                           OutputIterator first_to);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2>
auto rotate_copy(const ExecutionSpace& exespace,                             (3)
                 const ::Kokkos::View<DataType1, Properties1...>& source,
                 std::size_t n_location,
                 const ::Kokkos::View<DataType2, Properties2...>& dest);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2>
auto rotate_copy(const std::string& label, const ExecutionSpace& exespace,   (4)
                 const ::Kokkos::View<DataType1, Properties1...>& source,
                 std::size_t n_location,
                 const ::Kokkos::View<DataType2, Properties2...>& dest);

} //end namespace Experimental
} //end namespace Kokkos
```

## Description

Copies the elements from the range `[first_from, last_from)` to the range
starting at `first_to` or from `view_from` to `view_dest` in such a way that
the element `n_first` or `view(n_location)` becomes the first element of the
new range and `n_first - 1` becomes the last element.

## Parameters and Requirements

- `exespace`:
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::rotate_copy_iterator_api_default"
  - for 3, the default string is: "Kokkos::rotate_copy_view_api_default"
- `first_from, last_from`:
  - range of elements to copy from
  - must be *random access iterators*
  - must represent a valid range, i.e., `last_from >= first_from` (this condition is checked in debug mode)
  - must be accessible from `exespace`
- `first_to`:
  - beginning of the range to copy to
  - must be a *random access iterator*
  - must be accessible from `exespace`
- `n_first`:
  - iterator to element that should be the first of the rotated range
  - must be a *random access iterator*
  - must be such that `[first_from, n_first)` and `[n_first, last_from)` are valid ranges.
  - must be accessible from `exespace`
- `view_from, view_to`:
  - source and destination views
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`
- `n_location`:
  - integer value identifying the element to rotate about

## Return

Iterator to the element *after* the last element copied.
