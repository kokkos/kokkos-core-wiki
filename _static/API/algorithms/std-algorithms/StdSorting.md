# Sorting

Header File: `Kokkos_StdAlgorithms.hpp`

**All algorithms are currently in the `Kokkos::Experimental` namespace.**

Currently supported (see below the full details):
 - [`is_sorted`](#kokkosexperimentalis_sorted)
 - [`is_sorted_until`](#kokkosexperimentalis_sorted_until)

------

## `Kokkos::Experimental::is_sorted`

```c++
//
// overload set accepting iterators
//
template <class ExecutionSpace, class IteratorType>
bool is_sorted(const ExecutionSpace& exespace,                              (1)
               IteratorType first, IteratorType last);

template <class ExecutionSpace, class IteratorType>
bool is_sorted(const std::string& label, const ExecutionSpace& exespace,    (2)
               IteratorType first, IteratorType last);

template <class ExecutionSpace, class IteratorType, class ComparatorType>
bool is_sorted(const ExecutionSpace& exespace,                              (3)
               IteratorType first, IteratorType last,
               ComparatorType comp);

template <class ExecutionSpace, class IteratorType, class ComparatorType>
bool is_sorted(const std::string& label, const ExecutionSpace& exespace,    (4)
               IteratorType first, IteratorType last,
			   ComparatorType comp);

//
// overload set accepting views
//
template <class ExecutionSpace, class DataType, class... Properties>
bool is_sorted(const ExecutionSpace& exespace,                              (5)
               const ::Kokkos::View<DataType, Properties...>& view);

template <class ExecutionSpace, class DataType, class... Properties>
bool is_sorted(const std::string& label, const ExecutionSpace& exespace,    (6)
               const ::Kokkos::View<DataType, Properties...>& view);

template <
  class ExecutionSpace,
  class DataType, class... Properties, class ComparatorType>
bool is_sorted(const ExecutionSpace& exespace,                              (7)
               const ::Kokkos::View<DataType, Properties...>& view,
               ComparatorType comp);

template <
  class ExecutionSpace,
  class DataType, class... Properties, class ComparatorType>
bool is_sorted(const std::string& label, const ExecutionSpace& exespace,    (8)
               const ::Kokkos::View<DataType, Properties...>& view,
               ComparatorType comp);
```

### Description

- (1,2,5,6): checks if the elements in the range `[first, last)` (1,2)
  or in `view` (5,6) are sorted in non-descending order using `operator<` to compare two elements.

- (3,4,7,8): checks if the elements in the range `[first, last)` (3,4)
  or in `view` (7,8) are sorted in non-descending order using the binary functor `comp` to compare two elements.

### Parameters and Requirements

- `exespace`:
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1,3 the default string is: "Kokkos::is_sorted_iterator_api_default"
  - for 5,7 the default string is: "Kokkos::is_sorted_view_api_default"
- `first`, `last`:
  - range of elements to examine
  - must be *random access iterators*
  - must represent a valid range, i.e., `last >= first` (checked in debug mode)
  - must be accessible from `exespace`
- `view`:
  - Kokkos view to examine
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`
- `comp`:
  - *binary* functor returning `true`
  if the first argument is *less than* the second argument;
  `comp(a,b)` must be valid to be called from the execution space passed,
  and convertible to bool for every pair of arguments `a,b` of type
  `value_type`, where `value_type` is the value type of `IteratorType` (for 1,2,3,4)
  or the value type of `view` (for 5,6,7,8) and must not modify `a,b`.
  - must conform to:
  ```c++
  struct Comparator
  {
     KOKKOS_INLINE_FUNCTION
     bool operator()(const value_type & a, const value_type & b) const {
	   return /* true if a is less than b, based on your logic of "less than" */;
     }
  };
  ```

### Return

True if the elements are sorted in descending order.



------


## `Kokkos::Experimental::is_sorted_until`

```c++
//
// overload set accepting iterators
//
template <class ExecutionSpace, class IteratorType>
IteratorType is_sorted_until(const ExecutionSpace& exespace,                     (1)
                             IteratorType first, IteratorType last);

template <class ExecutionSpace, class IteratorType>
IteratorType is_sorted_until(const std::string& label,                           (2)
                             const ExecutionSpace& exespace,
						     IteratorType first, IteratorType last);

template <class ExecutionSpace, class IteratorType, class ComparatorType>
IteratorType is_sorted_until(const ExecutionSpace& exespace,                     (3)
                             IteratorType first, IteratorType last,
                             ComparatorType comp);

template <class ExecutionSpace, class IteratorType, class ComparatorType>
IteratorType is_sorted_until(const std::string& label,                           (4)
                             const ExecutionSpace& exespace,
                             IteratorType first, IteratorType last,
			                 ComparatorType comp);

//
// overload set accepting views
//
template <class ExecutionSpace, class DataType, class... Properties>
auto is_sorted_until(const ExecutionSpace& exespace,                             (5)
                     const ::Kokkos::View<DataType, Properties...>& view);

template <class ExecutionSpace, class DataType, class... Properties>
auto is_sorted_until(const std::string& label, const ExecutionSpace& exespace,   (6)
                     const ::Kokkos::View<DataType, Properties...>& view);

template <
  class ExecutionSpace,
  class DataType, class... Properties, class ComparatorType>
auto is_sorted_until(const ExecutionSpace& exespace,                             (7)
                     const ::Kokkos::View<DataType, Properties...>& view,
                     ComparatorType comp);

template <
  class ExecutionSpace,
  class DataType, class... Properties, class ComparatorType>
auto is_sorted_until(const std::string& label, const ExecutionSpace& exespace,   (8)
                     const ::Kokkos::View<DataType, Properties...>& view,
                     ComparatorType comp);
```

### Description

- (1,2,5,6): finds the largest range beginning at `first` (1,2) or
  at `Kokkos::Experimental::begin(view)` (5,6) in which the elements
  are sorted in non-descending order. Comparison between elements is done via `operator<`.

- (3,4,7,8): finds the largest range beginning at `first` (3,4) or
  at `Kokkos::Experimental::begin(view)` (7,8) in which the elements
  are sorted in non-descending order. Comparison between elements is done via
  the binary functor `comp`.

### Parameters and Requirements

- `exespace`, `first`, `last`, `view`, `comp`: same as in [`is_sorted`](#kokkosexperimentalis_sorted)
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1,3 the default string is: "Kokkos::is_sorted_until_iterator_api_default"
  - for 5,7 the default string is: "Kokkos::is_sorted_until_view_api_default"
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`

### Return

- (1,2,3,4): the last iterator `it` for which range `[first, it)` is sorted,
  and where the following is true: `std::is_same_v<decltype(it), IteratorType>`.

- (5,6,7,8): the last iterator `it` for which range `[Kokkos::Experimental::begin(view), it)` is sorted.
  Note that `it` is computed as: `Kokkos::Experimental::begin(view) + increment` where `increment` is
  found in the algoritm.
