
# PartitioningOperations

Header File: `Kokkos_Core.hpp`

**All algorithms are currently in the `Kokkos::Experimental` namespace.**


## Kokkos::Experimental::is_partitioned

```cpp
template <class ExecutionSpace, class InputIterator, class PredicateType>
InputIterator is_partitioned(const ExecutionSpace& exespace,                              (1)
                             InputIterator first, InputIterator last,
                             PredicateType pred);

template <class ExecutionSpace, class InputIterator, class PredicateType>
InputIterator is_partitioned(const std::string& label, const ExecutionSpace& exespace,    (2)
                             InputIterator first, InputIterator last,
                             PredicateType pred);

template <class ExecutionSpace, class DataType, class... Properties, class PredicateType>
auto is_partitioned(const ExecutionSpace& exespace,
                    const ::Kokkos::View<DataType, Properties...>& view,                  (3)
                    PredicateType pred);

template <class ExecutionSpace, class DataType, class... Properties, class PredicateType>
auto is_partitioned(const std::string& label, const ExecutionSpace& exespace,
                    const ::Kokkos::View<DataType, Properties...>& view,                  (4)
                    PredicateType pred);
```

### Description

Returns `true` if all elements in `[first, last)` satisfying the predicate `pred` appear *before* all elements that don't, or if `[first, last)` is empty.

Overload set: (1,2) accept iterators; (3,4) accepting views


### Parameters and Requirements

- `exespace`, `label`, `first`, `last`, `view`, `pred`: same requirements as `find_if` (TODO: link)
- `label`:
  - for 1, the default string is: `Kokkos::is_partitioned_iterator_api_default`
  - for 3, the default string is: `Kokkos::is_partitioned_view_api_default`

### Return

- `true`: if range is partitioned according to `pred` or if range is empty
- `false`: otherwise

### Example
```cpp
namespace KE = Kokkos::Experimental;

template<class ValueType>
struct IsNegative
{
  KOKKOS_INLINE_FUNCTION
  bool operator()(const ValueType & operand) const {
    constexpr auto zero = static_cast<ValueType>(0);
    return (operand < zero);
  }
};

using view_type = Kokkos::View<int*>;
view_type a("a", 15);
// fill a somehow

auto exespace  = Kokkos::DefaultExecutionSpace;
const auto res = KE::is_partitioned(exespace, KE::cbegin(a), KE::cend(a), IsNegative<int>());
// note that we used {cbegin, cend} here, because checking if the view
// is partitioned does not need to have write access to the view's elements.
// We could have also used {begin, end}, but if we know that we only
// need read access, it is good practice to ensure const-correctness.

// ...
```
