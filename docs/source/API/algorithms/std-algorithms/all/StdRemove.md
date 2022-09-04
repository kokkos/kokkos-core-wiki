
# `remove`

Header File: `Kokkos_StdAlgorithms.hpp`

```c++
namespace Kokkos{
namespace Experimental{

template <class ExecutionSpace, class Iterator, class ValueType>
Iterator remove(const ExecutionSpace& exespace,                             (1)
                Iterator first, Iterator last,
                const ValueType& value);

template <class ExecutionSpace, class Iterator, class ValueType>
Iterator remove(const std::string& label,                                   (2)
                const ExecutionSpace& exespace,
                Iterator first, Iterator last,
                const ValueType& value);

template <
  class ExecutionSpace,
  class DataType, class... Properties,
  class ValueType>
auto remove(const ExecutionSpace& exespace,                                 (3)
            const ::Kokkos::View<DataType, Properties...>& view,
            const ValueType& value);

template <
  class ExecutionSpace,
  class DataType, class... Properties,
  class ValueType>
auto remove(const std::string& label,                                       (4)
            const ExecutionSpace& exespace,
            const ::Kokkos::View<DataType, Properties...>& view,
            const ValueType& value);

} //end namespace Experimental
} //end namespace Kokkos
```

## Description

Removes all elements equal to `value` by shifting via move assignment
the elements in the range `[first, last)` (1,2) or in `view` (3,4)
such that the elements not to be removed
appear in the beginning of the range (1,2) or in the beginning of `view` (3,4).
Relative order of the elements that remain is preserved
and the physical size of the container is unchanged.

## Parameters and Requirements

- `exespace`:
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::remove_iterator_api_default"
  - for 3, the default string is: "Kokkos::remove_view_api_default"
- `first, last`:
  - range of elements to modify
  - must be *random access iterators*
  - must represent a valid range, i.e., `last >= first` (checked in debug mode)
  - must be accessible from `exespace`
- `view`:
  - view of elements to modify
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`
- `value`:
  - target value to remove

## Return

Iterator to the element *after* the new logical end.
