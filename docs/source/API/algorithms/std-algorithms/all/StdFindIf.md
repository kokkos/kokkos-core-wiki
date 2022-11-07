
# `find_if`

Header File: `Kokkos_StdAlgorithms.hpp`

```cpp
namespace Kokkos{
namespace Experimental{

template <class ExecutionSpace, class InputIterator, class PredicateType>
InputIterator find_if(const ExecutionSpace& exespace,                                (1)
                      InputIterator first, InputIterator last,
                      PredicateType pred);

template <class ExecutionSpace, class InputIterator, class PredicateType>
InputIterator find_if(const std::string& label, const ExecutionSpace& exespace,      (2)
                      InputIterator first, InputIterator last,
                      PredicateType pred);

template <class ExecutionSpace, class DataType, class... Properties, class PredicateType>
auto find_if(const ExecutionSpace& exespace,
             const Kokkos::View<DataType, Properties...>& view,                      (3)
             PredicateType pred);

template <class ExecutionSpace, class DataType, class... Properties, class PredicateType>
auto find_if(const std::string& label, const ExecutionSpace& exespace,
             const Kokkos::View<DataType, Properties...>& view,                      (4)
             PredicateType pred);

} //end namespace Experimental
} //end namespace Kokkos
```

## Description

Returns an iterator to the *first* element in `[first, last)` (1,2) or `view` (3,4) 
for which the predicate returns `true`.

## Parameters and Requirements

- `exespace`, `first, last`, `view`: same as in [`find`](./StdFind)

- `label`:
  - for 1, the default string is: "Kokkos::find_if_iterator_api_default"
  - for 3, the default string is: "Kokkos::find_if_view_api_default"

- `pred`:
  - unary predicate which returns `true` for the required element; `pred(v)` must be valid to be called from the execution space passed, and convertible to bool for every
 argument `v` of type (possible const) `value_type`, where `value_type` is the value type of `InputIterator`, and must not modify `v`.
  - must conform to:
  ```cpp
  struct Predicate
  {
     KOKKOS_INLINE_FUNCTION
     bool operator()(const /*type needed */ & operand) const { return /* ... */; }

     // or, also valid

     KOKKOS_INLINE_FUNCTION
     bool operator()(/*type needed */ operand) const { return /* ... */; }
  };
  ```

## Return

- (1,2): `InputIterator` instance pointing to the first element where the predicate is evaluated to true, or `last` if no such element is found
- (3,4): iterator to the first element where the predicate is evaluated to `true`, 
or `Kokkos::Experimental::end(view)` if no such element is found

## Example

```cpp
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

auto exespace = Kokkos::DefaultExecutionSpace;
using view_type = Kokkos::View<exespace, int*>;
view_type a("a", 15);
// fill "a" somehow

// create predicate
EqualsValue<int> p(5);

auto it1 = KE::find_if(exespace, KE::begin(a), KE::end(a), p);

// assuming OpenMP is enabled, then you can also explicitly call
auto it2 = KE::find_if(Kokkos::OpenMP(), KE::begin(a), KE::end(a), p);
```
