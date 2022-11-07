
# `find_end`

Header File: `Kokkos_StdAlgorithms.hpp`

```cpp
namespace Kokkos{
namespace Experimental{

template <class ExecutionSpace, class IteratorType1, class IteratorType2>
IteratorType1 find_end(const ExecutionSpace& exespace, IteratorType1 first,
                       IteratorType1 last, IteratorType2 s_first,                     (1)
                       IteratorType2 s_last);

template <class ExecutionSpace, class IteratorType1, class IteratorType2>
IteratorType1 find_end(const std::string& label, const ExecutionSpace& exespace,
                       IteratorType1 first, IteratorType1 last,                       (2)
                       IteratorType2 s_first, IteratorType2 s_last);

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2>
auto find_end(const ExecutionSpace& exespace,
              const ::Kokkos::View<DataType1, Properties1...>& view,                  (3)
              const ::Kokkos::View<DataType2, Properties2...>& s_view);

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2>
auto find_end(const std::string& label, const ExecutionSpace& exespace,
              const ::Kokkos::View<DataType1, Properties1...>& view,                  (4)
              const ::Kokkos::View<DataType2, Properties2...>& s_view);

// overload set 2: binary predicate passed
template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class BinaryPredicateType>
IteratorType1 find_end(const ExecutionSpace& exespace, IteratorType1 first,
                       IteratorType1 last, IteratorType2 s_first,                     (5)
                       IteratorType2 s_last, const BinaryPredicateType& pred);

template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class BinaryPredicateType>
IteratorType1 find_end(const std::string& label, const ExecutionSpace& exespace,
                       IteratorType1 first, IteratorType1 last,                       (6)
                       IteratorType2 s_first, IteratorType2 s_last,
                       const BinaryPredicateType& pred);

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class BinaryPredicateType>
auto find_end(const ExecutionSpace& exespace,
              const ::Kokkos::View<DataType1, Properties1...>& view,                  (7)
              const ::Kokkos::View<DataType2, Properties2...>& s_view,
              const BinaryPredicateType& pred);

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class BinaryPredicateType>
auto find_end(const std::string& label, const ExecutionSpace& exespace,
              const ::Kokkos::View<DataType1, Properties1...>& view,                  (8)
              const ::Kokkos::View<DataType2, Properties2...>& s_view,
              const BinaryPredicateType& pred);

} //end namespace Experimental
} //end namespace Kokkos
```

## Description

- 1,2,5,6: searches for the last occurrence of the sequence `[s_first, s_last)` 
in the range `[first, last)` comparing elements via `operator ==` (1,2) or via `pred` (5,6)

- 3,4,7,8: searches for the last occurrence of the `s_view` in `view` 
comparing elements via `operator ==` (3,4 or via `pred` (7,8)

## Parameters and Requirements

- `exespace`, `s_first`, `s_last`, `first`, `last`, `s_view` and `view` similar to [`search`](./StdSearch).

- `label`:
    - 1,5: The default string is "Kokkos::find_end_iterator_api_default".
    - 3,7: The default string is "Kokkos::find_end_view_api_default".

- `pred` - similar to [`equal`](./StdEqual)
