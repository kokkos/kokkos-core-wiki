# Minimum/maximum

Header File: `Kokkos_StdAlgorithms.hpp`

**All algorithms are currently in the `Kokkos::Experimental` namespace.**

Currently supported (see below the full details):
 - [`min_element`](#kokkosexperimentalmin_element)
 - [`max_element`](#kokkosexperimentalmax_element)
 - [`minmax_element`](#kokkosexperimentalminmax_element)

------

## `Kokkos::Experimental::min_element`

```c++
//
// overload set accepting iterators
//
template <class ExecutionSpace, class IteratorType>
auto min_element(const ExecutionSpace& exespace,                        (1)
                 IteratorType first, IteratorType last);

template <class ExecutionSpace, class IteratorType>
auto min_element(const std::string& label,                              (2)
                 const ExecutionSpace& exespace,
                 IteratorType first, IteratorType last);

template <class ExecutionSpace, class IteratorType, class ComparatorType>
auto min_element(const ExecutionSpace& exespace,                        (3)
                 IteratorType first, IteratorType last,
                 ComparatorType comp);

template <class ExecutionSpace, class IteratorType, class ComparatorType>
auto min_element(const std::string& label,                              (4)
                 const ExecutionSpace& exespace,
                 IteratorType first, IteratorType last,
                 ComparatorType comp);

//
// overload set accepting views
//
template <class ExecutionSpace, class DataType, class... Properties>
auto min_element(const ExecutionSpace& exespace,                        (5)
                 const ::Kokkos::View<DataType, Properties...>& view);

template <class ExecutionSpace, class DataType, class... Properties>
auto min_element(const std::string& label,                              (6)
                 const ExecutionSpace& exespace,
                 const ::Kokkos::View<DataType, Properties...>& view);

template <class ExecutionSpace, class DataType, class ComparatorType, class... Properties>
auto min_element(const ExecutionSpace& exespace,                        (7)
                 const ::Kokkos::View<DataType, Properties...>& view,
                 ComparatorType comp);

template <class ExecutionSpace, class DataType, class ComparatorType, class... Properties>
auto min_element(const std::string& label,                              (8)
                 const ExecutionSpace& exespace,
                 const ::Kokkos::View<DataType, Properties...>& view,
                 ComparatorType comp);
```

### Description

- (1,2,5,6): Finds the smallest element in the range `[first, last)` (1,2)
  or in `view` (5,6) using `operator<` to compare two elements

- (3,4,7,8): Finds the smallest element in the range `[first, last)` (3,4)
  or in `view` (7,8) using the binary functor `comp` to compare two elements

### Parameters and Requirements

- `exespace`:
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1,3 the default string is: "Kokkos::min_element_iterator_api_default"
  - for 5,7 the default string is: "Kokkos::min_element_view_api_default"
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

     KOKKOS_INLINE_FUNCTION
     bool operator()(const volatile value_type a, const volatile value_type b) const {
     return /* true if a is less than b, based on your logic of "less than" */;
     }
  };
  ```
  - the volatile overload is needed because the algorithm is
  currently implemented as a reduction, where the `comp` functor is used
  as the ``joiner'' to join two values. The current Kokkos implementation
  of reductions requires any custom joiner to have
  a volatile overload: see [Custom Reductions](../../../../ProgrammingGuide/Custom-Reductions) for more info on reductions.

### Return

Iterator to the smallest element.
The following special cases apply:

- if several elements are equivalent to the smallest element, it returns the iterator to the *first* such element.

- if the range `[first, last)` is empty it returns `last`.

- if `view` is empty, it returns `Kokkos::Experimental::end(view)`.

### Example

```c++
namespace KE = Kokkos::Experimental;
Kokkos::View<double*> a("a", 13);
// fill a somehow

auto res = KE::min_element(Kokkos::DefaultExecutionSpace(), KE::begin(a), KE::end(a));

// passing the view directly
auto res = KE::min_element(Kokkos::DefaultExecutionSpace(), a);


// using a custom comparator
template <class ValueType1, class ValueType2 = ValueType1>
struct CustomLessThanComparator {
  KOKKOS_INLINE_FUNCTION
  bool operator()(const ValueType1& a,
                  const ValueType2& b) const {
   // here we use < but one can put any custom logic to return true if a is less than b
    return a < b;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator()(const volatile ValueType1& a,
                  const volatile ValueType1& b) const {
    // here we use < but one can put any custom logic to return true if a is less than b
    return a < b;
  }

  KOKKOS_INLINE_FUNCTION
  CustomLessThanComparator() {}
};

// passing the view directly
auto res = KE::min_element(Kokkos::DefaultExecutionSpace(), a, CustomLessThanComparator<double>());
```


------


## `Kokkos::Experimental::max_element`

```c++
//
// overload set accepting iterators
//
template <class ExecutionSpace, class IteratorType>
auto max_element(const ExecutionSpace& exespace,                        (1)
                 IteratorType first, IteratorType last);

template <class ExecutionSpace, class IteratorType>
auto max_element(const std::string& label,                              (2)
                 const ExecutionSpace& exespace,
                 IteratorType first, IteratorType last);

template <class ExecutionSpace, class IteratorType, class ComparatorType>
auto max_element(const ExecutionSpace& exespace,                        (3)
                 IteratorType first, IteratorType last,
                 ComparatorType comp);

template <class ExecutionSpace, class IteratorType, class ComparatorType>
auto max_element(const std::string& label,                              (4)
                 const ExecutionSpace& exespace,
                 IteratorType first, IteratorType last,
                 ComparatorType comp);

//
// overload set accepting views
//
template <class ExecutionSpace, class DataType, class... Properties>
auto max_element(const ExecutionSpace& exespace,                        (5)
                 const ::Kokkos::View<DataType, Properties...>& view);

template <class ExecutionSpace, class DataType, class... Properties>
auto max_element(const std::string& label,                              (6)
                 const ExecutionSpace& exespace,
                 const ::Kokkos::View<DataType, Properties...>& view);

template <class ExecutionSpace, class DataType, class ComparatorType, class... Properties>
auto max_element(const ExecutionSpace& exespace,                        (7)
                 const ::Kokkos::View<DataType, Properties...>& view,
                 ComparatorType comp);

template <class ExecutionSpace, class DataType, class ComparatorType, class... Properties>
auto max_element(const std::string& label,                              (8)
                 const ExecutionSpace& exespace,
                 const ::Kokkos::View<DataType, Properties...>& view,
                 ComparatorType comp);
```

### Description

- (1,2,5,6): Finds the largest element in the range `[first, last)` or in `view`
  using `operator<` to compare two elements

- (3,4,7,8): Finds the largest element in the range `[first, last)` or in `view`
  using the binary functor `comp` to compare two elements

### Parameters and Requirements

- `exespace`, `first`, `last`, `view`, `comp`: same as in [`min_element`](#kokkosexperimentalmin_element)
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1,3 the default string is: "Kokkos::max_element_iterator_api_default"
  - for 5,7 the default string is: "Kokkos::max_element_view_api_default"

### Return

Iterator to the largest element.
The following special cases apply:
- if several elements are equivalent to the largest element, returns the iterator to the *first* such element.

- if the range `[first, last)` is empty it returns `last`.

- if `view` is empty, it returns `Kokkos::Experimental::end(view)`.


### Example

```c++
namespace KE = Kokkos::Experimental;
Kokkos::View<double*> a("a", 13);
// fill a somehow

auto res = KE::max_element(Kokkos::DefaultExecutionSpace(), KE::begin(a), KE::end(a));

// passing the view directly
auto res = KE::max_element(Kokkos::DefaultExecutionSpace(), a);


// using a custom comparator
template <class ValueType1, class ValueType2 = ValueType1>
struct CustomLessThanComparator {
  KOKKOS_INLINE_FUNCTION
  bool operator()(const ValueType1& a,
                  const ValueType2& b) const {
    // here we use < but one can put any custom logic to return true if a is less than b
    return a < b;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator()(const volatile ValueType1& a,
                  const volatile ValueType1& b) const {
    // here we use < but one can put any custom logic to return true if a is less than b
    return a < b;
  }

  KOKKOS_INLINE_FUNCTION
  CustomLessThanComparator() {}
};

// passing the view directly
auto res = KE::max_element(Kokkos::DefaultExecutionSpace(), a, CustomLessThanComparator<double>());
```


------


## `Kokkos::Experimental::minmax_element`

```c++
//
// overload set accepting iterators
//
template <class ExecutionSpace, class IteratorType>
auto minmax_element(const ExecutionSpace& exespace,                        (1)
                    IteratorType first, IteratorType last);

template <class ExecutionSpace, class IteratorType>
auto minmax_element(const std::string& label,                              (2)
                    const ExecutionSpace& exespace,
                    IteratorType first, IteratorType last);

template <class ExecutionSpace, class IteratorType, class ComparatorType>
auto minmax_element(const ExecutionSpace& exespace,                        (3)
                    IteratorType first, IteratorType last,
                    ComparatorType comp);

template <class ExecutionSpace, class IteratorType, class ComparatorType>
auto minmax_element(const std::string& label,                              (4)
                    const ExecutionSpace& exespace,
                    IteratorType first, IteratorType last,
                    ComparatorType comp);

//
// overload set accepting views
//
template <class ExecutionSpace, class DataType, class... Properties>
auto minmax_element(const ExecutionSpace& exespace,                        (5)
                    const ::Kokkos::View<DataType, Properties...>& view);

template <class ExecutionSpace, class DataType, class... Properties>
auto minmax_element(const std::string& label,                              (6)
                    const ExecutionSpace& exespace,
                    const ::Kokkos::View<DataType, Properties...>& view);

template <class ExecutionSpace, class DataType, class ComparatorType, class... Properties>
auto minmax_element(const ExecutionSpace& exespace,                        (7)
                    const ::Kokkos::View<DataType, Properties...>& view,
                    ComparatorType comp);

template <class ExecutionSpace, class DataType, class ComparatorType, class... Properties>
auto minmax_element(const std::string& label,                              (8)
                    const ExecutionSpace& exespace,
                    const ::Kokkos::View<DataType, Properties...>& view,
                    ComparatorType comp);
```

### Description

- (1,2,5,6): Finds the smallest and largest elements in the
  range `[first, last)` (1,2) or in `view` (5,6) using `operator<` to compare two elements

- (3,4,7,8): Finds the smallest and largest element in the
  range `[first, last)` (3,4) or in `view` (7,8) using
  the binary functor `comp` to compare two elements

### Parameters and Requirements

- `exespace`, `first`, `last`, `view`, `comp`: same as in [`min_element`](#kokkosexperimentalmin_element)
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1,3 the default string is: "Kokkos::minmax_element_iterator_api_default"
  - for 5,7 the default string is: "Kokkos::minmax_element_view_api_default"

### Return

A Kokkos pair of iterators to the smallest and largest elements in that order.
The following special cases apply:

- if the range `[first, last)` is empty it returns `Kokkos::pair(first, first)`.

- if `view` is empty, it returns `Kokkos::pair(Kokkos::Experimental::begin(view), Kokkos::Experimental::begin(view))`.

- if several elements are equivalent to the smallest element,
  the iterator to the *first* such element is returned.

- if several elements are equivalent to the largest element,
  the iterator to the *last* such element is returned.


### Example

```c++
namespace KE = Kokkos::Experimental;
Kokkos::View<double*> a("a", 11);

auto itPair = KE::minmax_element(Kokkos::DefaultExecutionSpace(), KE::begin(a), KE::end(a));

// passing the view directly
auto itPair = KE::minmax_element(Kokkos::DefaultExecutionSpace(), a);


// using a custom comparator
template <class ValueType1, class ValueType2 = ValueType1>
struct CustomLessThanComparator {
  KOKKOS_INLINE_FUNCTION
  bool operator()(const ValueType1& a,
                  const ValueType2& b) const {
    // here we use < but one can put any custom logic to return true if a is less than b
    return a < b;
  }

  KOKKOS_INLINE_FUNCTION
  bool operator()(const volatile ValueType1& a,
                  const volatile ValueType1& b) const {
    // here we use < but one can put any custom logic to return true if a is less than b
    return a < b;
  }

  KOKKOS_INLINE_FUNCTION
  CustomLessThanComparator() {}
};

// passing the view directly
auto res = KE::minmax_element(Kokkos::DefaultExecutionSpace(), a, CustomLessThanComparator<double>());
```
