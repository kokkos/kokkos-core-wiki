
# `is_sorted`

Header File: `Kokkos_StdAlgorithms.hpp`

```c++
namespace Kokkos{
namespace Experimental{

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

} //end namespace Experimental
} //end namespace Kokkos
```

## Description

- (1,2,5,6): checks if the elements in the range `[first, last)` (1,2)
  or in `view` (5,6) are sorted in non-descending order using `operator<` to compare two elements.

- (3,4,7,8): checks if the elements in the range `[first, last)` (3,4)
  or in `view` (7,8) are sorted in non-descending order using the binary functor `comp` to compare two elements.

## Parameters and Requirements

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

## Return

True if the elements are sorted in descending order.