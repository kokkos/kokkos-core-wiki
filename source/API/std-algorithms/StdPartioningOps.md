
# Partitioning

Header File: `Kokkos_Core.hpp`

**All algorithms are currently in the `Kokkos::Experimental` namespace.**

Currently supported (see below the full details):
 - [`is_partitioned`](#kokkosexperimentalis_partitioned)
 - [`partition_copy`](#kokkosexperimentalpartition_copy)
 - [`partition_point`](#kokkosexperimentalpartition_point)


------


## `Kokkos::Experimental::is_partitioned`

```cpp
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
```

### Description

Returns `true` if all elements in the range `[first, last)` (1,2) or in
`view` (3,4) satisfying the predicate `pred` appear *before* all elements that don't.
If `[first, last)` or `view` is empty, returns `true`.

### Parameters and Requirements

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
  ```cpp
  struct Predicate
  {
     KOKKOS_INLINE_FUNCTION
     bool operator()(const value_type & v) const { return /* ... */; }

     // or, also valid

     KOKKOS_INLINE_FUNCTION
     bool operator()(value_type v) const { return /* ... */; }
  };
  ```

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
```

--------


## `Kokkos::Experimental::partition_point`

```cpp
template <class ExecutionSpace, class IteratorType, class PredicateType>
IteratorType partition_point(const ExecutionSpace& exespace,                   (1)
                             IteratorType first, IteratorType last,
                             PredicateType pred);

template <class ExecutionSpace, class IteratorType, class PredicateType>
IteratorType partition_point(const std::string& label,                         (2)
                             const ExecutionSpace& exespace,
                             IteratorType first, IteratorType last,
                             PredicateType pred);

template <class ExecutionSpace, class DataType, class... Properties, class PredicateType>
auto partition_point(const ExecutionSpace& exespace,
                     const ::Kokkos::View<DataType, Properties...>& view,      (3)
                     PredicateType pred);

template <class ExecutionSpace, class DataType, class... Properties, class PredicateType>
auto partition_point(const std::string& label,                                 (4)
                     const ExecutionSpace& exespace,
                     const ::Kokkos::View<DataType, Properties...>& view,
                     PredicateType pred);
```

### Description

Examines the range `[first, last)` or `view` and locates the
first element that does not satisfy `pred`.

Assumes the range (or the view) already to be partitioned.


### Parameters and Requirements

- `exespace`, `first`, `last`, `view`, `pred`: same as in [`is_partioned`](#kokkosexperimentalis_partitioned)
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::partition_point_iterator_api_default"
  - for 3, the default string is: "Kokkos::partition_point_view_api_default"

### Return

Iterator to the elment *after* the last element in the first partition,
or `last` if all elements satisfy `pred`.


--------


## `Kokkos::Experimental::partition_copy`

```cpp
template <
  class ExecutionSpace,
  class InputIteratorType,
  class OutputIteratorTrueType,
  class OutputIteratorFalseType,
  class PredicateType>
::Kokkos::pair<OutputIteratorTrueType, OutputIteratorFalseType>
partition_copy(const ExecutionSpace& exespace,                                  (1)
               InputIteratorType from_first,
			   InputIteratorType from_last,
			   OutputIteratorTrueType to_first_true,
			   OutputIteratorFalseType to_first_false,
			   PredicateType p);

template <
  class ExecutionSpace,
  class InputIteratorType,
  class OutputIteratorTrueType,
  class OutputIteratorFalseType,
  class PredicateType>
::Kokkos::pair<OutputIteratorTrueType, OutputIteratorFalseType>
partition_copy(const std::string& label,                                        (2)
               const ExecutionSpace& exespace,
               InputIteratorType from_first,
               InputIteratorType from_last,
               OutputIteratorTrueType to_first_true,
               OutputIteratorFalseType to_first_false,
               PredicateType p);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class DataType3, class... Properties3,
  class PredicateType>
auto partition_copy(const ExecutionSpace& exespace,                             (3)
                    const ::Kokkos::View<DataType1, Properties1...>& view_from,
                    const ::Kokkos::View<DataType2, Properties2...>& view_dest_true,
                    const ::Kokkos::View<DataType3, Properties3...>& view_dest_false,
                    PredicateType p);


template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class DataType3, class... Properties3,
  class PredicateType>
auto partition_copy(const std::string& label,                                   (4)
                    const ExecutionSpace& exespace,
                    const ::Kokkos::View<DataType1, Properties1...>& view_from,
                    const ::Kokkos::View<DataType2, Properties2...>& view_dest_true,
                    const ::Kokkos::View<DataType3, Properties3...>& view_dest_false,
                    PredicateType p);
```

### Description

- (1,2): copies from the range `[from_first, from_last)` the elements
  that satisfy the predicate `pred` to the range beginning at `to_first_true`,
  while the others are copied to the range beginning at `to_first_false`.

- (3,4): copies from `view` the elements that satisfy the predicate
  `pred` to `view_true`, while the others are copied to `view_false`.

### Parameters and Requirements

- `exespace`:
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::partition_copy_iterator_api_default"
  - for 3, the default string is: "Kokkos::partition_copy_view_api_default"
- `from_first, from_last`:
  - range of elements to copy from
  - must be *random access iterators*
  - must represent a valid range, i.e., `from_last >= from_first` (checked in debug mode)
  - must be accessible from `exespace`
- `to_first_true`:
  - beginning of the range to copy the elements that satisfy `pred` to
  - must be a *random access iterator*
  - must be accessible from `exespace`
- `to_first_false`:
  - beginning of the range to copy the elements that do NOT satisfy `pred` to
  - must be a *random access iterator*
  - must be accessible from `exespace`
- `view_from`:
  - source view of elements to copy from
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`
- `view_dest_true`:
  - destination view to copy the elements that satisfy `pred` to
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`
- `view_dest_false`:
  - destination view to copy the elements that do NOT satisfy `pred` to
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`
- `pred`:
  - *unary* predicate returning `true` for the required element to replace; `pred(v)`
  must be valid to be called from the execution space passed, and convertible to bool for every
  argument `v` of type (possible const) `value_type`, where `value_type`
  is the value type of `InputIteratorType` (for 1,2) or the value type of `view_from` (for 3,4),
  and must not modify `v`.
  - must conform to:
  ```cpp
  struct Predicate
  {
     KOKKOS_INLINE_FUNCTION
     bool operator()(const value_type & v) const { return /* ... */; }

     // or, also valid

     KOKKOS_INLINE_FUNCTION
     bool operator()(value_type v) const { return /* ... */; }
  };
  ```

### Return

A `Kokkos::pair` containing the iterators to the end of two destination ranges (or views)
