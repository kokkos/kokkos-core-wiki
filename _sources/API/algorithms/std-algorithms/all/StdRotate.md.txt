
# `rotate`

Header File: `Kokkos_StdAlgorithms.hpp`

```c++
namespace Kokkos{
namespace Experimental{

template <class ExecutionSpace, class IteratorType>
IteratorType rotate(const ExecutionSpace& exespace,                            (1)
                    IteratorType first,
                    IteratorType n_first,
                    IteratorType last);

template <class ExecutionSpace, class IteratorType>
IteratorType rotate(const std::string& label, const ExecutionSpace& exespace,  (2)
                    IteratorType first,
                    IteratorType n_first,
                    IteratorType last);

template <class ExecutionSpace, class DataType, class... Properties>
auto rotate(const ExecutionSpace& exespace,                                    (3)
            const ::Kokkos::View<DataType, Properties...>& view,
            std::size_t n_location);

template <class ExecutionSpace, class DataType, class... Properties>
auto rotate(const std::string& label, const ExecutionSpace& exespace,          (4)
            const ::Kokkos::View<DataType, Properties...>& view,
            std::size_t n_location);

} //end namespace Experimental
} //end namespace Kokkos
```

## Description

Swaps the elements in the range `[first, last)` or in `view` in such a way that
the element `n_first` or `view(n_location)` becomes the first element of the
new range and `n_first - 1` becomes the last element.


## Parameters and Requirements

- `exespace`:
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::rotate_iterator_api_default"
  - for 3, the default string is: "Kokkos::rotate_view_api_default"
- `first, last`:
  - range of elements to modify
  - must be *random access iterators*
  - must represent a valid range, i.e., `last >= first` (checked in debug mode)
  - must be accessible from `exespace`
- `n_first`:
  - iterator to element that should be the first of the rotated range
  - must be a *random access iterator*
  - must be such that `[first, n_first)` and `[n_first, last)` are valid ranges.
  - must be accessible from `exespace`
- `view`:
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`
- `n_location`:
  - integer value identifying the element to rotate about


## Return

- For (1,2), returns the iterator computed as `first + (last - n_first)`

- For (3,4), returns `Kokkos::begin(view) + (Kokkos::end(view) - n_location)`
