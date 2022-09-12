# `max_element`

Header File: `Kokkos_StdAlgorithms.hpp`

```c++
namespace Kokkos{
namespace Experimental{

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

} //end namespace Experimental
} //end namespace Kokkos
```

## Description

- (1,2,5,6): Finds the largest element in the range `[first, last)` or in `view`
  using `operator<` to compare two elements

- (3,4,7,8): Finds the largest element in the range `[first, last)` or in `view`
  using the binary functor `comp` to compare two elements

## Parameters and Requirements

- `exespace`, `first`, `last`, `view`, `comp`: same as in [`min_element`](./StdMinElement)
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1,3 the default string is: "Kokkos::max_element_iterator_api_default"
  - for 5,7 the default string is: "Kokkos::max_element_view_api_default"

## Return

Iterator to the largest element.
The following special cases apply:
- if several elements are equivalent to the largest element, returns the iterator to the *first* such element.

- if the range `[first, last)` is empty it returns `last`.

- if `view` is empty, it returns `Kokkos::Experimental::end(view)`.


## Example

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
  CustomLessThanComparator() {}
};

// passing the view directly
auto res = KE::max_element(Kokkos::DefaultExecutionSpace(), a, CustomLessThanComparator<double>());
```
