
# `remove_copy`

Header File: `Kokkos_StdAlgorithms.hpp`

```c++
namespace Kokkos{
namespace Experimental{

template <
  class ExecutionSpace,
  class InputIterator, class OutputIterator,
  class ValueType>
OutputIterator remove_copy(const ExecutionSpace& exespace,                   (1)
                           InputIterator first_from,
                           InputIterator last_from,
                           OutputIterator first_to,
                           const ValueType& value);

template <
  class ExecutionSpace,
  class InputIterator, class OutputIterator,
  class ValueType>
OutputIterator remove_copy(const std::string& label,                         (2)
                           const ExecutionSpace& exespace,
                           InputIterator first_from,
                           InputIterator last_from,
                           OutputIterator first_to,
                           const ValueType& value);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class ValueType>
auto remove_copy(const ExecutionSpace& exespace,                             (3)
                 const ::Kokkos::View<DataType1, Properties1...>& view_from,
                 const ::Kokkos::View<DataType2, Properties2...>& view_dest,
                 const ValueType& value);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class ValueType>
auto remove_copy(const std::string& label,                                   (4)
                 const ExecutionSpace& exespace,
                 const ::Kokkos::View<DataType1, Properties1...>& view_from,
                 const ::Kokkos::View<DataType2, Properties2...>& view_dest,
                 const ValueType& value);

} //end namespace Experimental
} //end namespace Kokkos
```

## Description

Copies the elements from ther range `[first_from, last_from)` to a new
range starting at `first_to` or from `view_from` to `view_dest` omitting
those that are equal to `value`.


## Parameters and Requirements

- `exespace`:
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::remove_copy_iterator_api_default"
  - for 3, the default string is: "Kokkos::remove_copy_view_api_default"
- `first_from, last_from`:
  - range of elements to copy from
  - must be *random access iterators*
  - must represent a valid range, i.e., `last_from >= first_from` (checked in debug mode)
  - must be accessible from `exespace`
- `first_to`:
  - beginning of the range to copy to
  - must be *random access iterators*
  - must be accessible from `exespace`
- `view_from`, `view_dest`:
  - source and destination views
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`
- `value`:
  - target value to omit

## Return

Iterator to the element after the last element copied.
