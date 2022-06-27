# NonModifying Sequence

Header File: `Kokkos_StdAlgorithms.hpp`

**All algorithms are currently in the `Kokkos::Experimental` namespace.**

MISSING SOME HERE

## Kokkos::Experimental::find

```c++
template <class ExecutionSpace, class InputIterator, class T>
InputIterator find(const ExecutionSpace& exespace,                              (1)
                   InputIterator first, InputIterator last,
                   const T& value);

template <class ExecutionSpace, class InputIterator, class T>
InputIterator find(const std::string& label, const ExecutionSpace& exespace,    (2)
                   InputIterator first, InputIterator last,
                   const T& value);

template <class ExecutionSpace, class DataType, class... Properties, class T>
auto find(const ExecutionSpace& exespace,                                       (3)
          const ::Kokkos::View<DataType, Properties...>& view,
          const T& value);

template <class ExecutionSpace, class DataType, class... Properties, class T>
auto find(const std::string& label, const ExecutionSpace& exespace,             (4)
          const ::Kokkos::View<DataType, Properties...>& view,
          const T& value);
```

### Description

Returns an iterator to the *first* element in `[first, last)` that equals `value`. Equality is checked using `operator==`.

- (1,2): overload set accepting iterators
- (3,4): overload set accepting views

### Parameters and Requirements

- `exespace`:
  - execution space instance
- `label`:
  - string forwarded to all implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::find_iterator_api_default"
  - for 3, the default string is: "Kokkos::find_view_api_default"
- `first, last`:
  - range of elements to search in
  - must be *random access iterators*, e.g., returned from `Kokkos::Experimental::(c)begin/(c)end`
  - must represent a valid range, i.e., `last >= first` (this condition is checked in debug mode)
  - must be accessible from `exespace`
- `view`:
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`

### Return

- (1,2): `InputIterator` instance pointing to the first element that equals `value`, or `last` if no elements is found
- (2,3): iterator to the first element that equals `value`, or `Kokkos::Experimental::end(view)` if none is found

### Example
```c++
namespace KE = Kokkos::Experimental;
using view_type = Kokkos::View<int*>;
view_type a("a", 15);
// fill a somehow

auto exespace = Kokkos::DefaultExecutionSpace;
auto it1 = KE::find(exespace, KE::cbegin(a), KE::cend(a), 5);

// assuming OpenMP is enabled and "a" is host-accessible, you can also do
auto it2 = KE::find(Kokkos::OpenMP(), KE::begin(a), KE::end(a), 5);
```



## Kokkos::Experimental::find_if

```c++
template <class ExecutionSpace, class InputIterator, class PredicateType>
InputIterator find_if(const ExecutionSpace& exespace,                              (1)
                      InputIterator first, InputIterator last,
                      PredicateType pred);

template <class ExecutionSpace, class InputIterator, class PredicateType>
InputIterator find_if(const std::string& label, const ExecutionSpace& exespace,    (2)
                      InputIterator first, InputIterator last,
                      PredicateType pred);

template <class ExecutionSpace, class DataType, class... Properties, class PredicateType>
auto find_if(const ExecutionSpace& exespace,
             const ::Kokkos::View<DataType, Properties...>& view,                  (3)
             PredicateType pred);

template <class ExecutionSpace, class DataType, class... Properties, class PredicateType>
auto find_if(const std::string& label, const ExecutionSpace& exespace,
             const ::Kokkos::View<DataType, Properties...>& view,                  (4)
             PredicateType pred);
```

### Description

Returns an iterator to the *first* element in `[first, last)` for which the predicate returns `true`.

- (1,2): overload set accepting iterators
- (3,4): overload set accepting views

### Parameters and Requirements

- `exespace`, `first, last`, `view`: same as in `find` (TODO: link)
- `label`:
  - for 1, the default string is: "Kokkos::find_if_iterator_api_default"
  - for 3, the default string is: "Kokkos::find_if_view_api_default"
- `pred`:
  - unary predicate which returns `true` for the required element; `pred(v)` must be valid to be called from the execution space passed, and convertible to bool for every
 argument `v` of type (possible const) `value_type`, where `value_type` is the value type of `InputIterator`, and must not modify `v`.
  - must conform to:
  ```c++
  struct Predicate
  {
     KOKKOS_INLINE_FUNCTION
     bool operator()(const /*type needed */ & operand) const { return /* ... */; }

     // or, also valid

     KOKKOS_INLINE_FUNCTION
     bool operator()(/*type needed */ operand) const { return /* ... */; }
  };
  ```


### Return

- (1,2): `InputIterator` instance pointing to the first element satisfying condition, or `last` if no elements is found
- (2,3): iterator to the first element that equals `value`, or `Kokkos::Experimental::end(view)` if none is found

### Example
```c++
namespace KE = Kokkos::Experimental;

template<class ValueType>
struct EqualsValue
{
  const ValueType m_value;
  EqualsValFunctor(ValueType value) : m_value(value){}

  KOKKOS_INLINE_FUNCTION
  bool operator()(const ValueType & operand) const {
    return operand == m_value;
  }
};

using view_type = Kokkos::View<int*>;
view_type a("a", 15);
// fill a somehow

// create predicate
EqualsValue<int> p(5);

auto exespace = Kokkos::DefaultExecutionSpace;
auto it1 = KE::find_if(exespace, KE::begin(a), KE::end(a), p);

// assuming OpenMP is enabled, then you can also explicitly call
auto it2 = KE::find_if(Kokkos::OpenMP(), KE::begin(a), KE::end(a), p);
```
