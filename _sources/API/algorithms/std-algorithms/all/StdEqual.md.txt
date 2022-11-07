
# `equal`

Header File: `Kokkos_StdAlgorithms.hpp`

```cpp
namespace Kokkos{
namespace Experimental{

template <class ExecutionSpace, class IteratorType1, class IteratorType2>
bool equal(const ExecutionSpace& exespace, IteratorType1 first1, 
           IteratorType1 last1, IteratorType2 first2);                              (1)

template <class ExecutionSpace, class IteratorType1, class IteratorType2>
bool equal(const std::string& label, const ExecutionSpace& exespace,
           IteratorType1 first1, IteratorType1 last1, IteratorType2 first2);        (2)

template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class BinaryPredicateType>
bool equal(const ExecutionSpace& exespace, IteratorType1 first1,                    (3)
           IteratorType1 last1, IteratorType2 first2, 
           BinaryPredicateType predicate);

template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class BinaryPredicateType>
bool equal(const std::string& label, const ExecutionSpace& exespace,
           IteratorType1 first1, IteratorType1 last1, IteratorType2 first2,         (4)
           BinaryPredicateType predicate);

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2>
bool equal(const ExecutionSpace& exespace,
           const Kokkos::View<DataType1, Properties1...>& view1,                    (5)
           Kokkos::View<DataType2, Properties2...>& view2);

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2>
bool equal(const std::string& label, const ExecutionSpace& exespace,
           const Kokkos::View<DataType1, Properties1...>& view1,                    (6)
           Kokkos::View<DataType2, Properties2...>& view2);

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class BinaryPredicate>
bool equal(const ExecutionSpace& exespace,
           const Kokkos::View<DataType1, Properties1...>& view1,                    (7)
           Kokkos::View<DataType2, Properties2...>& view2, BinaryPredicate pred);

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class BinaryPredicate>
bool equal(const std::string& label, const ExecutionSpace& exespace,
           const Kokkos::View<DataType1, Properties1...>& view1,                    (8)
           Kokkos::View<DataType2, Properties2...>& view2, BinaryPredicate pred);

} //end namespace Experimental
} //end namespace Kokkos
```

## Description

- (1,2,3,4): returns true if the range `[first1, last1)` is equal to the range `[first2, first2 + (last1 - first1))`, 
and false otherwise
- (5,6,7,8): returns true if `view1` and `view2` are equal and false otherwise

- for (1,2,5,6) equality is checked via `operator == `, while for (3,4,7,8) equality is checked via 
the binary predicate `pred`.

## Parameters and Requirements

- `exespace`:
  - execution space instance

- `label`:
  - (1,3): The default string is "Kokkos::equal_iterator_api_default"
  - (5,7): The default string is "Kokkos::equal_view_api_default"

- `first1`, `last1`, `first2`:
  - range of elements to read and compare
  - must be *random access iterators*
  - must represent a valid range, i.e., `last1 >= first1` (checked in debug mode)
  - must be accessible from `exespace`

- `view1`, `view2`:
  - views to read elements and compare
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`

- `pred`
  ```cpp
  template <class ValueType1, class ValueType2 = ValueType1>
  struct IsEqualFunctor {
      KOKKOS_INLINE_FUNCTION
      bool operator()(const ValueType1& a, const ValueType2& b) const {
        return (a == b);
      }
  };
  ```

## Return

- `true` or `false` for (1,2,5,6) based on `operator == `.
- `true` or `false` for (3,4,7,8) based on the BinaryPredicate `pred`

## Example

```cpp
namespace KE = Kokkos::Experimental;

template <class ValueType1, class ValueType2 = ValueType1>
struct IsEqualFunctor {

  KOKKOS_INLINE_FUNCTION
  bool operator()(const ValueType1& a, const ValueType2& b) const {
    return (a == b);
  }
};

auto exespace = Kokkos::DefaultExecutionSpace;
using view_type = Kokkos::View<exespace, int*>;
view_type a("a", 15);
view_type b("b", 15);
// fill a,b somehow

// create functor
IsEqualFunctor<int,int> p();

bool isEqual = KE::equal(exespace, KE::begin(a), KE::end(a), KE::begin(b), KE::end(b) p);

// assuming OpenMP is enabled, then you can also explicitly call
bool isEqual = KE::equal(Kokkos::OpenMP(), KE::begin(a), KE::end(a), KE::begin(b), KE::end(b), p);
```
