
# `find_if_not`

Header File: `Kokkos_StdAlgorithms.hpp`

```cpp
namespace Kokkos{
namespace Experimental{

template <class ExecutionSpace, class InputIterator, class PredicateType>
InputIterator find_if_not(const ExecutionSpace& exespace,                            (1)
                      InputIterator first, InputIterator last,
                      PredicateType pred);

template <class ExecutionSpace, class InputIterator, class PredicateType>
InputIterator find_if_not(const std::string& label, const ExecutionSpace& exespace,  (2)
                      InputIterator first, InputIterator last,
                      PredicateType pred);

template <class ExecutionSpace, class DataType, class... Properties, class PredicateType>
auto find_if_not(const ExecutionSpace& exespace,
             const Kokkos::View<DataType, Properties...>& view,                      (3)
             PredicateType pred);

template <class ExecutionSpace, class DataType, class... Properties, class PredicateType>
auto find_if_not(const std::string& label, const ExecutionSpace& exespace,
             const Kokkos::View<DataType, Properties...>& view,                      (4)
             PredicateType pred);

} //end namespace Experimental
} //end namespace Kokkos
```

## Description

Returns an iterator to the *first* element in `[first, last)` (1,2) or `view` (3,4) 
for which the predicate returns `false`.

## Parameters and Requirements

- `exespace`, `first, last`, `view`, `pred`: same as in [`find_if`](./StdFindIf)

- `label`:
  - for 1, the default string is: "Kokkos::find_if_not_iterator_api_default"
  - for 3, the default string is: "Kokkos::find_if_not_view_api_default"
