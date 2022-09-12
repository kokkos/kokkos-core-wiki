
# `minmax_element`

Header File: `Kokkos_StdAlgorithms.hpp`

```c++
namespace Kokkos{ 
namespace Experimental{

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

} //end namespace Experimental
} //end namespace Kokkos
```

## Description

- (1,2,5,6): Finds the smallest and largest elements in the
  range `[first, last)` (1,2) or in `view` (5,6) using `operator<` to compare two elements

- (3,4,7,8): Finds the smallest and largest element in the
  range `[first, last)` (3,4) or in `view` (7,8) using
  the binary functor `comp` to compare two elements

## Parameters and Requirements

- `exespace`, `first`, `last`, `view`, `comp`: same as in [`min_element`](./StdMinElement)
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1,3 the default string is: "Kokkos::minmax_element_iterator_api_default"
  - for 5,7 the default string is: "Kokkos::minmax_element_view_api_default"

## Return

A Kokkos pair of iterators to the smallest and largest elements in that order.
The following special cases apply:

- if the range `[first, last)` is empty it returns `Kokkos::pair(first, first)`.

- if `view` is empty, it returns `Kokkos::pair(Kokkos::Experimental::begin(view), Kokkos::Experimental::begin(view))`.

- if several elements are equivalent to the smallest element,
  the iterator to the *first* such element is returned.

- if several elements are equivalent to the largest element,
  the iterator to the *last* such element is returned.


## Example

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
  CustomLessThanComparator() {}
};

// passing the view directly
auto res = KE::minmax_element(Kokkos::DefaultExecutionSpace(), a, CustomLessThanComparator<double>());
```
