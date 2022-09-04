
# `reverse`

Header File: `Kokkos_StdAlgorithms.hpp`

```c++
namespace Kokkos{
namespace Experimental{

template <class ExecutionSpace, class IteratorType, class T>
void reverse(const ExecutionSpace& exespace,                                    (1)
             IteratorType first, IteratorType last);

template <class ExecutionSpace, class IteratorType, class T>
void reverse(const std::string& label, const ExecutionSpace& exespace,          (2)
             IteratorType first, IteratorType last);

template <class ExecutionSpace, class DataType, class... Properties, class T>
void reverse(const ExecutionSpace& exespace,                                    (3)
             const Kokkos::View<DataType, Properties...>& view);

template <class ExecutionSpace, class DataType, class... Properties, class T>
void reverse(const std::string& label, const ExecutionSpace& exespace,          (4)
             const Kokkos::View<DataType, Properties...>& view);

} //end namespace Experimental
} //end namespace Kokkos
```

## Description

Reverses ther order of the elements in the range `[first, last)` (overloads 1,2)
or in `view` (overloads 3,4).


## Parameters and Requirements

- `exespace`:
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::reverse_iterator_api_default"
  - for 3, the default string is: "Kokkos::reverse_view_api_default"
- `first, last`:
  - range of elements to reverse
  - must be *random access iterators*
  - must represent a valid range, i.e., `last >= first` (this condition is checked in debug mode)
  - must be accessible from `exespace`
- `view`:
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`


## Return

None