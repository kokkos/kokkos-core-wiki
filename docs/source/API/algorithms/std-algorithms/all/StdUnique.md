
# `unique`

Header File: `Kokkos_StdAlgorithms.hpp`

```c++
namespace Kokkos{
namespace Experimental{

template <class ExecutionSpace, class IteratorType>
IteratorType unique(const ExecutionSpace& exespace,                          (1)
                    IteratorType first, IteratorType last);

template <class ExecutionSpace, class IteratorType>
IteratorType unique(const std::string& label,                                (2)
                    const ExecutionSpace& exespace,
                    IteratorType first, IteratorType last);

template <class ExecutionSpace, class DataType, class... Properties>
auto unique(const ExecutionSpace& exespace,                                  (3)
            const Kokkos::View<DataType, Properties...>& view);

template <class ExecutionSpace, class DataType, class... Properties>
auto unique(const std::string& label, const ExecutionSpace& exespace,        (4)
            const Kokkos::View<DataType, Properties...>& view);

template <class ExecutionSpace, class IteratorType, class BinaryPredicate>
IteratorType unique(const ExecutionSpace& exespace,                          (5)
                    IteratorType first, IteratorType last,
                    BinaryPredicate pred);

template <class ExecutionSpace, class IteratorType, class BinaryPredicate>
IteratorType unique(const std::string& label,                                (6)
                    const ExecutionSpace& exespace,
                    IteratorType first, IteratorType last,
                    BinaryPredicate pred);

template <
  class ExecutionSpace,
  class DataType, class... Properties,
  class BinaryPredicate>
auto unique(const ExecutionSpace& exespace,                                  (7)
            const Kokkos::View<DataType, Properties...>& view,
            BinaryPredicate pred);

template <
  class ExecutionSpace,
  class DataType, class... Properties,
  class BinaryPredicate>
auto unique(const std::string& label,                                        (8)
            const ExecutionSpace& exespace,
            const Kokkos::View<DataType, Properties...>& view,
            BinaryPredicate pred);

} //end namespace Experimental
} //end namespace Kokkos
```

## Description

- Overloads 1,2,5,6: eliminates all except the first element from every
consecutive group of equivalent elements from the range `[first, last)`,
and returns an iterator to the element *after* the new logical end of the range.
Equivalence is checked using `operator==` for (1,2) and the binary predicate `pred` for (5,6).

- Overloads 3,4,7,8: eliminates all except the first element from every
consecutive group of equivalent elements in `view`, and returns an
iterator to the element *after* the new logical end of the range.
Equivalence is checked using `operator==` for (3,4) and the binary predicate `pred` for (7,8).

## Parameters and Requirements

- `exespace`:
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::unique_iterator_api_default"
  - for 3, the default string is: "Kokkos::unique_ranges_view_api_default"
- `first, last`:
  - iterators to range to examine
  - must be *random access iterators*
  - must represent a valid range, i.e., `last >= first` (checked in debug mode)
  - must be accessible from `exespace`
- `view`:
  - view to examine, must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`

## Return

Iterator to the element *after* the new logical end of the range
