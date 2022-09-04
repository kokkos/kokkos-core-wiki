
# `copy`

Header File: `Kokkos_StdAlgorithms.hpp`

```c++
namespace Kokkos{
namespace Experimental{

template <class ExecutionSpace, class InputIteratorType, class OutputIteratorType>
OutputIteratorType copy(const ExecutionSpace& exespace,                      (1)
                        InputIteratorType first_from,
                        InputIteratorType last_from,
                        OutputIteratorType first_to);

template <class ExecutionSpace, class InputIteratorType, class OutputIteratorType>
OutputIteratorType copy(const std::string& label,                            (2)
                        const ExecutionSpace& exespace,
                        InputIteratorType first_from,
                        InputIteratorType last_from,
                        OutputIteratorType first_to);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2
>
auto copy(const ExecutionSpace& exespace,                                    (3)
          const Kokkos::View<DataType1, Properties1...>& view_from,
          const Kokkos::View<DataType2, Properties2...>& view_to);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2
>
auto copy(const std::string& label, const ExecutionSpace& exespace,          (4)
          const Kokkos::View<DataType1, Properties1...>& view_from,
          const Kokkos::View<DataType2, Properties2...>& view_to);

} //end namespace Experimental
} //end namespace Kokkos
```

## Description

Copies the elements from range `[first_from, last_from)` to another
range beginning at `first_to` (overloads 1,2) or from
a source view `view_from` to a destination view `view_to` (overloads 3,4).

## Parameters and Requirements

- `exespace`:
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::copy_iterator_api_default"
  - for 3, the default string is: "Kokkos::copy_view_api_default"
- `first_from, last_from`:
  - range of elements to copy from
  - must be *random access iterators*
  - must represent a valid range, i.e., `last_from >= first_from` (checked in debug mode)
  - must be accessible from `exespace`
- `first_to`:
  - beginning of the range to copy to
  - must be a *random access iterator*
  - must be accessible from `exespace`
- `view_from`, `view_to`:
  - source and destination views to copy elements from and to
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`


## Return

Iterator to the destination element *after* the last element copied.
