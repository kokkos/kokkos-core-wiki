
# `is_partitioned`

Header File: `Kokkos_StdAlgorithms.hpp`

```c++
namespace Kokkos{
namespace Experimental{

template <class ExecutionSpace, class InputIterator, class PredicateType>
bool is_partitioned(const ExecutionSpace& exespace,                              (1)
                    InputIterator first, InputIterator last,
                    PredicateType pred);

template <class ExecutionSpace, class InputIterator, class PredicateType>
bool is_partitioned(const std::string& label, const ExecutionSpace& exespace,    (2)
                    InputIterator first, InputIterator last,
                    PredicateType pred);

template <class ExecutionSpace, class DataType, class... Properties, class PredicateType>
auto is_partitioned(const ExecutionSpace& exespace,
                    const ::Kokkos::View<DataType, Properties...>& view,         (3)
                    PredicateType pred);

template <class ExecutionSpace, class DataType, class... Properties, class PredicateType>
auto is_partitioned(const std::string& label, const ExecutionSpace& exespace,
                    const ::Kokkos::View<DataType, Properties...>& view,         (4)
                    PredicateType pred);

} //end namespace Experimental
} //end namespace Kokkos
```

## Description

Returns `true` if all elements in the range `[first, last)` (1,2) or in
`view` (3,4) satisfying the predicate `pred` appear *before* all elements that don't.
If `[first, last)` or `view` is empty, returns `true`.

## Parameters and Requirements

- `exespace`:
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::is_partitioned_iterator_api_default"
  - for 3, the default string is: "Kokkos::is_partitioned_view_api_default"
- `first, last`:
  - range of elements to search in
  - must be *random access iterators*, e.g., `Kokkos::Experimental::begin/end`
  - must represent a valid range, i.e., `last >= first` (this condition is checked in debug mode)
  - must be accessible from `exespace`
- `view`:
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`
- `pred`:
  - *unary* predicate returning `true` for the required element to replace; `pred(v)`
  must be valid to be called from the execution space passed, and convertible to bool for every
  argument `v` of type (possible const) `value_type`, where `value_type`
  is the value type of `IteratorType` (for 1,2) or the value type of `view` (for 3,4),
  and must not modify `v`.
  - must conform to:
  ```c++
  struct Predicate
  {
     KOKKOS_INLINE_FUNCTION
     bool operator()(const value_type & v) const { return /* ... */; }

     // or, also valid

     KOKKOS_INLINE_FUNCTION
     bool operator()(value_type v) const { return /* ... */; }
  };
  ```

## Return

- `true`: if range is partitioned according to `pred` or if range is empty
- `false`: otherwise

## Example
```c++
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
```
