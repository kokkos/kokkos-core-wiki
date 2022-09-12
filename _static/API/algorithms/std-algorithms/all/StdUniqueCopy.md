
# `unique_copy`

Header File: `Kokkos_StdAlgorithms.hpp`

```c++
namespace Kokkos{
namespace Experimental{

template <class ExecutionSpace, class InputIterator, class OutputIterator>
OutputIterator unique_copy(const ExecutionSpace& exespace,                    (1)
                           InputIterator first_from, InputIterator last_from,
                           OutputIterator first_to);

template <class ExecutionSpace, class InputIterator, class OutputIterator>
OutputIterator unique_copy(const std::string& label,                          (2)
                           const ExecutionSpace& exespace,
                           InputIterator first_from, InputIterator last_from,
                           OutputIterator first_to);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2>
auto unique_copy(const ExecutionSpace& exespace,                              (3)
                 const Kokkos::View<DataType1, Properties1...>& source,
                 const Kokkos::View<DataType2, Properties2...>& dest);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2>
auto unique_copy(const std::string& label,                                    (4)
                 const ExecutionSpace& exespace,
                 const Kokkos::View<DataType1, Properties1...>& source,
                 const Kokkos::View<DataType2, Properties2...>& dest);

template <
  class ExecutionSpace,
  class InputIterator, class OutputIterator,
  class BinaryPredicate>
OutputIterator unique_copy(const ExecutionSpace& exespace,                    (5)
                           InputIterator first_from, InputIterator last_from,
                           OutputIterator first_to,
                           BinaryPredicate pred);

template <
  class ExecutionSpace,
  class InputIterator, class OutputIterator,
  class BinaryPredicate>
OutputIterator unique_copy(const std::string& label,                          (6)
                           const ExecutionSpace& exespace,
                           InputIterator first_from, InputIterator last_from,
                           OutputIterator first_to,
                           BinaryPredicate pred);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class BinaryPredicate>
auto unique_copy(const ExecutionSpace& exespace,                              (7)
                 const Kokkos::View<DataType1, Properties1...>& source,
                 const Kokkos::View<DataType2, Properties2...>& dest,
                 BinaryPredicate pred);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class BinaryPredicate>
auto unique_copy(const std::string& label,                                    (8)
                 const ExecutionSpace& exespace,
                 const Kokkos::View<DataType1, Properties1...>& source,
                 const Kokkos::View<DataType2, Properties2...>& dest,
                 BinaryPredicate pred);

} //end namespace Experimental
} //end namespace Kokkos
```

## Description

- Overloads 1,2,5,6: copies the elements from the range `[first_from, last_from)`
to a range starting at `first_to` such that there are no consecutive equal elements.
It returns an iterator to the element *after* the last element copied in the destination.
Equivalence is checked using `operator==` for (1,2) and the binary predicate `pred` for (5,6).

- Overloads 3,4,7,8: copies the elements from the `source` view to the `dest` view
such that there are no consecutive equal elements.
It returns an iterator to the element *after* the last element copied in the destination view.
Equivalence is checked using `operator==` for (3,4) and the binary predicate `pred` for (7,8).

## Parameters and Requirements

- `exespace`:
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::unique_copy_iterator_api_default"
  - for 3, the default string is: "Kokkos::unique_copy_ranges_view_api_default"
- `first_from, last_from`, `first_to`:
  - iterators to source range `{first,last}_from` and destination range `first_to`
  - must be *random access iterators*
  - must represent a valid range, i.e., `last_from >= first_from` (checked in debug mode)
  - must be accessible from `exespace`
- `source`, `dest`:
  - source and destination views, must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`

## Return

Iterator to the element *after* the last element copied in the destination range or view
