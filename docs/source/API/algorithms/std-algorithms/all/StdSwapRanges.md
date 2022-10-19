# `swap_ranges`

Header File: `Kokkos_StdAlgorithms.hpp`

```c++
namespace Kokkos{
namespace Experimental{

template <class ExecutionSpace, class Iterator1, class Iterator2>
Iterator2 swap_ranges(const ExecutionSpace& exespace,                        (1)
                      Iterator1 first1, Iterator1 last1,
                      Iterator2 first2);

template <class ExecutionSpace, class Iterator1, class Iterator2>
Iterator2 swap_ranges(const std::string& label,                              (2)
                      const ExecutionSpace& exespace,
                      Iterator1 first1, Iterator1 last1,
                      Iterator2 first2);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2
>
auto swap_ranges(const ExecutionSpace& exespace,                             (3)
                 const Kokkos::View<DataType1, Properties1...>& source,
                 Kokkos::View<DataType2, Properties2...>& dest);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2
>
auto swap_ranges(const std::string& label,                                   (4)
                 const ExecutionSpace& exespace,
                 const Kokkos::View<DataType1, Properties1...>& source,
                 Kokkos::View<DataType2, Properties2...>& dest);

} //end namespace Experimental
} //end namespace Kokkos
```

## Description

- Overloads 1,2: swaps the elements between the range `[first1, last1)`
and the range beginning at `first2`

- Overloads 3,4: swaps the elements between `source` and `dest` view


## Parameters and Requirements

- `exespace`:
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::swap_ranges_iterator_api_default"
  - for 3, the default string is: "Kokkos::swap_ranges_view_api_default"
- `first1, last1`, `first2`:
  - iterators to ranges to swap from and to
  - must be *random access iterators*
  - must represent a valid range, i.e., `last1 >= first1` (checked in debug mode)
  - must be accessible from `exespace`
- `source, dest`:
  - views to move from and to, must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`

## Return

Iterator to the element *after* the last element swapped.
