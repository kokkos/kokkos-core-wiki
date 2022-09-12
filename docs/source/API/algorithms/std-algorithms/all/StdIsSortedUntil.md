
# `is_sorted_until`

Header File: `Kokkos_StdAlgorithms.hpp`

```c++
namespace Kokkos{
namespace Experimental{

//
// overload set accepting iterators
//
template <class ExecutionSpace, class IteratorType>
IteratorType is_sorted_until(const ExecutionSpace& exespace,                     (1)
                             IteratorType first, IteratorType last);

template <class ExecutionSpace, class IteratorType>
IteratorType is_sorted_until(const std::string& label,                           (2)
                             const ExecutionSpace& exespace,
						     IteratorType first, IteratorType last);

template <class ExecutionSpace, class IteratorType, class ComparatorType>
IteratorType is_sorted_until(const ExecutionSpace& exespace,                     (3)
                             IteratorType first, IteratorType last,
                             ComparatorType comp);

template <class ExecutionSpace, class IteratorType, class ComparatorType>
IteratorType is_sorted_until(const std::string& label,                           (4)
                             const ExecutionSpace& exespace,
                             IteratorType first, IteratorType last,
			                 ComparatorType comp);

//
// overload set accepting views
//
template <class ExecutionSpace, class DataType, class... Properties>
auto is_sorted_until(const ExecutionSpace& exespace,                             (5)
                     const ::Kokkos::View<DataType, Properties...>& view);

template <class ExecutionSpace, class DataType, class... Properties>
auto is_sorted_until(const std::string& label, const ExecutionSpace& exespace,   (6)
                     const ::Kokkos::View<DataType, Properties...>& view);

template <
  class ExecutionSpace,
  class DataType, class... Properties, class ComparatorType>
auto is_sorted_until(const ExecutionSpace& exespace,                             (7)
                     const ::Kokkos::View<DataType, Properties...>& view,
                     ComparatorType comp);

template <
  class ExecutionSpace,
  class DataType, class... Properties, class ComparatorType>
auto is_sorted_until(const std::string& label, const ExecutionSpace& exespace,   (8)
                     const ::Kokkos::View<DataType, Properties...>& view,
                     ComparatorType comp);

} //end namespace Experimental
} //end namespace Kokkos
```

## Description

- (1,2,5,6): finds the largest range beginning at `first` (1,2) or
  at `Kokkos::Experimental::begin(view)` (5,6) in which the elements
  are sorted in non-descending order. Comparison between elements is done via `operator<`.

- (3,4,7,8): finds the largest range beginning at `first` (3,4) or
  at `Kokkos::Experimental::begin(view)` (7,8) in which the elements
  are sorted in non-descending order. Comparison between elements is done via
  the binary functor `comp`.

## Parameters and Requirements

- `exespace`, `first`, `last`, `view`, `comp`: same as in [`is_sorted`](./StdIsSorted)
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1,3 the default string is: "Kokkos::is_sorted_until_iterator_api_default"
  - for 5,7 the default string is: "Kokkos::is_sorted_until_view_api_default"
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`

## Return

- (1,2,3,4): the last iterator `it` for which range `[first, it)` is sorted,
  and where the following is true: `std::is_same_v<decltype(it), IteratorType>`.

- (5,6,7,8): the last iterator `it` for which range `[Kokkos::Experimental::begin(view), it)` is sorted.
  Note that `it` is computed as: `Kokkos::Experimental::begin(view) + increment` where `increment` is
  found in the algoritm.
