
# `reverse_copy`

Header File: `Kokkos_StdAlgorithms.hpp`

```c++
namespace Kokkos{
namespace Experimental{

template <class ExecutionSpace, class InputIterator, class OutputIterator>
OutputIterator reverse_copy(const ExecutionSpace& exespace,                  (1)
                            InputIterator first_from,
                            InputIterator last_from,
                            OutputIterator first_to);

template <class ExecutionSpace, class InputIterator, class OutputIterator>
OutputIterator reverse_copy(const std::string& label,                        (2)
                            const ExecutionSpace& exespace,
                            InputIterator first_from,
                            InputIterator last_from,
                            OutputIterator first_to);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2
>
auto reverse_copy(const ExecutionSpace& exespace,                            (3)
                  const Kokkos::View<DataType1, Properties1...>& source,
                  Kokkos::View<DataType2, Properties2...>& dest);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2
>
auto reverse_copy(const std::string& label,                                  (4)
                  const ExecutionSpace& exespace,
                  const Kokkos::View<DataType1, Properties1...>& source,
                  Kokkos::View<DataType2, Properties2...>& dest);

} //end namespace Experimental
} //end namespace Kokkos
```

## Description

- Overloads 1,2: copies the elements from the range `[first_from, last_from)`
and writes them in reverse order to the range beginning at `first_to`

- Overloads 3,4: copies the elements from `source` view and writes
them in reverse order to `dest` view


## Parameters and Requirements

- `exespace`:
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::reverse_copy_iterator_api_default"
  - for 3, the default string is: "Kokkos::reverse_copy_view_api_default"
- `first_from, last_from`:
  - range of elements to copy from
  - must be *random access iterators*
  - must represent a valid range, i.e., `last_from >= first_from` (checked in debug mode)
  - must be accessible from `exespace`
- `source, dest`:
  - views to copy from and write to, must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`


## Return

Iterator to the element *after* the last element copied.
