
# NonModifyingSequenceOperation

Header File: `Kokkos_Core.hpp`

**All algorithms are currently in the `Kokkos::Experimental` namespace.**

## Kokkos::Experimental::find

```cpp
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
```cpp
namespace KE = Kokkos::Experimental;
using view_type = Kokkos::View<int*>;
view_type a("a", 15);
// fill "a" somehow

auto exespace = Kokkos::DefaultExecutionSpace;
auto it1 = KE::find(exespace, KE::cbegin(a), KE::cend(a), 5);

// assuming OpenMP is enabled and "a" is host-accessible, you can also do
auto it2 = KE::find(Kokkos::OpenMP(), KE::begin(a), KE::end(a), 5);
```



## Kokkos::Experimental::find_if

```cpp
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
             const Kokkos::View<DataType, Properties...>& view,                    (3)
             PredicateType pred);

template <class ExecutionSpace, class DataType, class... Properties, class PredicateType>
auto find_if(const std::string& label, const ExecutionSpace& exespace,
             const Kokkos::View<DataType, Properties...>& view,                    (4)
             PredicateType pred);
```

### Description

Returns an iterator to the *first* element in `[first, last)` for which the predicate returns `true`.

- (1,2): overload set accepting iterators
- (3,4): overload set accepting views

### Parameters and Requirements

- `exespace`, `first, last`, `view`: same as in [`Kokkos::Experimental::find`](#Kokkos::Experimental::find)
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

### Return

- (1,2): `InputIterator` instance pointing to the first element where the predicate is evaluated to true, or `last` if no such element is found
- (3,4): iterator to the first element where the predicate is evaluated to `true`, or `Kokkos::Experimental::end(view)` if no such element is found

### Example
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
using view_type = Kokkos::View<execspace, int*>;
view_type a("a", 15);
// fill a somehow

// create predicate
EqualsValue<int> p(5);

auto it1 = KE::find_if(exespace, KE::begin(a), KE::end(a), p);

// assuming OpenMP is enabled, then you can also explicitly call
auto it2 = KE::find_if(Kokkos::OpenMP(), KE::begin(a), KE::end(a), p);
```


## Kokkos::Experimental::find_if_not

```cpp
template <class ExecutionSpace, class InputIterator, class PredicateType>
InputIterator find_if_not(const ExecutionSpace& exespace,                              (1)
                      InputIterator first, InputIterator last,
                      PredicateType pred);

template <class ExecutionSpace, class InputIterator, class PredicateType>
InputIterator find_if_not(const std::string& label, const ExecutionSpace& exespace,    (2)
                      InputIterator first, InputIterator last,
                      PredicateType pred);

template <class ExecutionSpace, class DataType, class... Properties, class PredicateType>
auto find_if_not(const ExecutionSpace& exespace,
             const Kokkos::View<DataType, Properties...>& view,                      (3)
             PredicateType pred);

template <class ExecutionSpace, class DataType, class... Properties, class PredicateType>
auto find_if_not(const std::string& label, const ExecutionSpace& exespace,
             const Kokkos::View<DataType, Properties...>& view,                      (4)
             PredicateType pred);
```

### Description

Returns an iterator to the *first* element in `[first, last)` for which the predicate returns `false`.

- (1,2): overload set accepting iterators
- (3,4): overload set accepting views

### Parameters and Requirements

- `exespace`, `first, last`, `view`, `pred`: same as in [Kokkos::Experimental::find_if](#KE::find_if)
- `label`:
  - for 1, the default string is: "Kokkos::find_if_not_iterator_api_default"
  - for 3, the default string is: "Kokkos::find_if_not_view_api_default"



## Kokkos::Experimental::for_each

```cpp
template <class ExecutionSpace, class InputIterator, class UnaryFunctorType>
UnaryFunctorType for_each(const ExecutionSpace& exespace,                              (1)
                      InputIterator first, InputIterator last,
                      UnaryFunctorType functor);

template <class ExecutionSpace, class InputIterator, class UnaryFunctorType>
UnaryFunctorType for_each(const std::string& label, const ExecutionSpace& exespace,    (2)
                      InputIterator first, InputIterator last,
                      UnaryFunctorType functor);

template <class ExecutionSpace, class DataType, class... Properties, class UnaryFunctorType>
UnaryFunctorType for_each(const ExecutionSpace& exespace,
             const Kokkos::View<DataType, Properties...>& view,                     (3)
             UnaryFunctorType functor);

template <class ExecutionSpace, class DataType, class... Properties, class PredicateType>
UnaryFunctorType for_each(const std::string& label, const ExecutionSpace& exespace,
             const Kokkos::View<DataType, Properties...>& view,                     (4)
             UnaryFunctorType func);
```

### Description

Applies the UnaryFunctorType `func` to the result of dereferencing each iterator in `[first,last)` for (1,2) and to the view elements in (3,4).

- (1,2): overload set accepting iterators
- (3,4): overload set accepting views

### Parameters and Requirements

- `exespace`, `first, last`, `view`: same as in [Kokkos::Experimental::find_if](#KE::find_if)
- `label`:
  - for 1, the default string is: "Kokkos::for_each_iterator_api_default"
  - for 3, the default string is: "Kokkos::for_each_view_api_default"
- `func`:
  - function object called on the all the elements;
  - The signature of the function should be `func(v)`.
  - Must be valid to be called from the execution space passed, and must accept every argument `v` of type (possible const) `value_type`, where `value_type` is the value type of `InputIterator`, and must not modify `v`.
  - must conform to:
  ```cpp
  struct func
  {
     KOKKOS_INLINE_FUNCTION
     void operator()(const /*type needed */ & operand) const { /* ... */; }

     // or, also valid

     KOKKOS_INLINE_FUNCTION
     void operator()(/*type needed */ operand) const { /* ... */; }
  };
  ```

### Return
UnaryFunctorType

### Example
```cpp
namespace KE = Kokkos::Experimental;

template<class ValueType>
struct IncrementValsFunctor
{
  const ValueType m_value;
  IncrementValsFunctor(ValueType value) : m_value(value){}

  KOKKOS_INLINE_FUNCTION
  void operator()(const ValueType & operand) const {
    operand += m_value;
  }
};

auto exespace = Kokkos::DefaultExecutionSpace;
using view_type = Kokkos::View<execspace, int*>;
view_type a("a", 15);
// fill "a" somehow

// create functor
IncrementValsFunctor<int> p(5);

// Increment each element in "a" by 5.
KE::for_each(exespace, KE::begin(a), KE::end(a), p);

// assuming OpenMP is enabled, then you can also explicitly call
KE::for_each(Kokkos::OpenMP(), KE::begin(a), KE::end(a), p);
```


## Kokkos::Experimental::for_each_n

```cpp
template <class ExecutionSpace, class InputIterator, class SizeType, class UnaryFunctorType>
UnaryFunctorType for_each_n(const ExecutionSpace& exespace,
                      InputIterator first, SizeType n,
                      UnaryFunctorType functor);                                   (1)

template <class ExecutionSpace, class InputIterator, class SizeType, class UnaryFunctorType>
UnaryFunctorType for_each_n(const std::string& label, const ExecutionSpace& exespace,
                      InputIterator first, SizeType n
                      UnaryFunctorType functor);                                   (2)

template <class ExecutionSpace, class DataType, class... Properties, class SizeType, class UnaryFunctorType>
UnaryFunctorType for_each_n(const ExecutionSpace& exespace,
             const Kokkos::View<DataType, Properties...>& view, SizeType n,
             UnaryFunctorType functor);                                            (3)

template <class ExecutionSpace, class DataType, class... Properties, class SizeType, class UnaryFunctorType>
UnaryFunctorType for_each_n(const std::string& label, const ExecutionSpace& exespace,
             const Kokkos::View<DataType, Properties...>& view, SizeType n,
             UnaryFunctorType func);                                               (4)
```

### Description

Applies the UnaryFunctorType `func` to the result of dereferencing each iterator in `[first,first+n]` for (1,2) and in (3,4) the functor is applied to the first `n` elements of the view.

- (1,2): overload set accepting iterators
- (3,4): overload set accepting views

### Parameters and Requirements

- `execspace`, `first`, `view`, `func` : same as in [Kokkos::Experimental::for_each](#KE::for_each)
- `label`:
  - for 1, the default string is: "Kokkos::for_each_n_iterator_api_default"
  - for 3, the default string is: "Kokkos::for_each_n_view_api_default"

### Return ###
UnaryFunctorType


## Kokkos::Experimental::mismatch

```cpp
template <class ExecutionSpace, class IteratorType1, class IteratorType2>
Kokkos::pair<IteratorType1, IteratorType2> mismatch(const ExecutionSpace& ex,
                  IteratorType1 first1,
                  IteratorType1 last1,                                             (1)
                  IteratorType2 first2,
                  IteratorType2 last2);

template <class ExecutionSpace, class IteratorType1, class IteratorType2>
Kokkos::pair<IteratorType1, IteratorType2> mismatch(
                const std::string& label,
                const ExecutionSpace& ex,
                IteratorType1 first1,
                IteratorType1 last1,                                               (2)
                IteratorType2 first2,
                IteratorType2 last2)

template <class ExecutionSpace, class IteratorType1, class IteratorType2, class BinaryPredicate>
Kokkos::pair<IteratorType1, IteratorType2> mismatch(const ExecutionSpace& ex,
                  IteratorType1 first1,
                  IteratorType1 last1,                                             (3)
                  IteratorType2 first2,
                  IteratorType2 last2, BinaryPredicate pred);

template <class ExecutionSpace, class IteratorType1, class IteratorType2, class BinaryPredicate>
Kokkos::pair<IteratorType1, IteratorType2> mismatch(const std::string& label,
                  const ExecutionSpace& ex,
                  IteratorType1 first1,
                  IteratorType1 last1,                                             (4)
                  IteratorType2 first2,
                  IteratorType2 last2, BinaryPredicate pred);


template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2>
auto mismatch(const ExecutionSpace& ex,
              const Kokkos::View<DataType1, Properties1...>& view1,                (5)
              const Kokkos::View<DataType2, Properties2...>& view2);

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2>
auto mismatch(const std::string& label, const ExecutionSpace& ex,
              const Kokkos::View<DataType1, Properties1...>& view1,                (6)
              const Kokkos::View<DataType2, Properties2...>& view2);

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class BinaryPredicateType>
auto mismatch(const ExecutionSpace& ex,
              const Kokkos::View<DataType1, Properties1...>& view1,                (7)
              const Kokkos::View<DataType2, Properties2...>& view2,
              BinaryPredicateType&& predicate);

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class BinaryPredicateType>
auto mismatch(const std::string& label, const ExecutionSpace& ex,
              const Kokkos::View<DataType1, Properties1...>& view1,                (8)
              const Kokkos::View<DataType2, Properties2...>& view2,
              BinaryPredicateType&& predicate);

```

### Description ###

Returns the first mismatching pair of elements from two ranges: one defined by [first1, last1) and another defined by [first2,last2) for (1,2,3,4).
Returns the first mismatching pair of elements from the two views `view1` and `view2` in (5,6,7,8).
The elements are compared using `operator==` in (1,2,5,6).
The elements in (3,4,7,8) are compared using a BinaryPredicate `pred`.

### Parameters and Requirements ###
- `execspace` and the InputIterators `first1`, `last1`, `first2`, `last2` similar to [Kokkos::Experimental::for_each](#kokkosexperimentalfor_each)
- `label`:
  - for 1,3, the default string is: "Kokkos::mismatch_iterator_api_default"
  - for 5,7, the default string is: "Kokkos::mismatch_view_api_default"
- `pred`
  ```cpp
  template <class ValueType1, class ValueType2 = ValueType1>
  struct IsEqualFunctor {

  KOKKOS_INLINE_FUNCTION
  Kokkos::pair<ValueType1, ValueType2> operator()(const ValueType1& a, const ValueType2& b) const {
    return (a == b);
    }
  };
 ```

### Return ###
- (1,2) - Kokkos::pair, where the `.first` and `.second` are the IteratorType1 and IteratorType2 instances where the `operator==` evaluates to false  
- (3,4) - Kokkos::pair, where the `.first` and `.second` are the IteratorType1 and IteratorType2 instances where the `pred` evaluates to false

### Example ###
```cpp
namespace KE = Kokkos::Experimental;

template <class ValueType1, class ValueType2 = ValueType1>
struct MismatchFunctor {

  KOKKOS_INLINE_FUNCTION
  Kokkos::pair<ValueType1, ValueType2> operator()(const ValueType1& a, const ValueType2& b) const {
    if(a != b)
        return (Kokkos::pair<ValueType1, ValueType2> (a,b));
  }
};

auto exespace = Kokkos::DefaultExecutionSpace;
using view_type = Kokkos::View<execspace, int*>;
view_type a("a", 15);
view_type b("b", 15);
// fill a,b somehow

// create functor
MisMatchFunctor<int, int> p();

Kokkos::pair<int,int> mismatch_index = KE::mismatch(exespace, KE::begin(a), KE::end(a), KE::begin(b), KE::end(b) p);

// assuming OpenMP is enabled, then you can also explicitly call
Kokkos::pair<int,int> mismatch_index = KE::mismatch(Kokkos::OpenMP(), KE::begin(a), KE::end(a), KE::begin(b), KE::end(b), p);

```


## Kokkos::Experimental::equal
```cpp
template <class ExecutionSpace, class IteratorType1, class IteratorType2>
bool equal(const ExecutionSpace& ex, IteratorType1 first1, IteratorType1 last1,                                             (1)
           IteratorType2 first2);

template <class ExecutionSpace, class IteratorType1, class IteratorType2>
bool equal(const std::string& label, const ExecutionSpace& ex,
           IteratorType1 first1, IteratorType1 last1, IteratorType2 first2);                                                (2)

template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class BinaryPredicateType>
bool equal(const ExecutionSpace& ex, IteratorType1 first1, IteratorType1 last1,                                             (3)
           IteratorType2 first2, BinaryPredicateType predicate);

template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class BinaryPredicateType>
bool equal(const std::string& label, const ExecutionSpace& ex,
           IteratorType1 first1, IteratorType1 last1, IteratorType2 first2,                                                 (4)
           BinaryPredicateType predicate);

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2>
bool equal(const ExecutionSpace& ex,
           const Kokkos::View<DataType1, Properties1...>& view1,                                                            (5)
           Kokkos::View<DataType2, Properties2...>& view2);

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2>
bool equal(const std::string& label, const ExecutionSpace& ex,
           const Kokkos::View<DataType1, Properties1...>& view1,                                                            (6)
           Kokkos::View<DataType2, Properties2...>& view2);

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class BinaryPredicate>
bool equal(const ExecutionSpace& ex,
           const Kokkos::View<DataType1, Properties1...>& view1,                                                            (7)
           Kokkos::View<DataType2, Properties2...>& view2, BinaryPredicate pred);

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class BinaryPredicate>
bool equal(const std::string& label, const ExecutionSpace& ex,
           const Kokkos::View<DataType1, Properties1...>& view1,                                                            (8)
           Kokkos::View<DataType2, Properties2...>& view2, BinaryPredicate pred);
```

### Description ###
Returns true if the range [first1, last1) is equal to the range [first2, first2 + (last1 - first1)), and false otherwise for (1,2,3,4).
Returns true if `view1` and `view2` are equal and false otherwise for (5,6,7,8)
- (1,2,5,6): Uses `operator == ` to compare the ranges
- (3,4,7,8): Uses binary predicate `pred` to compare the ranges.

### Parameters and Requirements ###
- `execspace`, the InputIterators `first1`, `last1`, `first2` and the views `view1`, `view2` similar to [Kokkos::Experimental::mismatch](#kokkosexperimentalmismatch)
- `label`:
  - (1,3): The default string is "Kokkos::equal_iterator_api_default"
  - (5,7): The default string is "Kokkos::equal_view_api_default"
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

### Return ###
- `true` or `false` for (1,2,5,6) based on `operator == `.
- `true` or `false` for (3,4,7,8) based on the BinaryPredicate `pred`

### Eample ###
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
using view_type = Kokkos::View<execspace, int*>;
view_type a("a", 15);
view_type b("b", 15);
// fill a,b somehow

// create functor
IsEqualFunctor<int,int> p();

bool isEqual = KE::equal(exespace, KE::begin(a), KE::end(a), KE::begin(b), KE::end(b) p);

// assuming OpenMP is enabled, then you can also explicitly call
bool isEqual = KE::equal(Kokkos::OpenMP(), KE::begin(a), KE::end(a), KE::begin(b), KE::end(b), p);

```

## Kokkos::Experimental::count_if

```cpp
template <class ExecutionSpace, class IteratorType, class Predicate>
typename IteratorType::difference_type count_if(const ExecutionSpace& ex,
                                                IteratorType first,
                                                IteratorType last,                                             (1)
                                                Predicate pred);
                                                

template <class ExecutionSpace, class IteratorType, class Predicate>
typename IteratorType::difference_type count_if(const std::string& label,
                                                const ExecutionSpace& ex,
                                                IteratorType first,                                            (2)
                                                IteratorType last,
                                                Predicate pred);
                                                
template <class ExecutionSpace, class DataType, class... Properties,
          class Predicate>
auto count_if(const ExecutionSpace& ex,
              const ::Kokkos::View<DataType, Properties...>& view,                                             (3)
              Predicate pred);
                                                
template <class ExecutionSpace, class DataType, class... Properties,
          class Predicate>
auto count_if(const std::string& label, const ExecutionSpace& ex,
              const ::Kokkos::View<DataType, Properties...>& view,                                             (4)
              Predicate pred);
```

### Description
Returns the number of elements in the range [first,last) for which `pred` is true in (1,2). 
For (3,4) returns the number of elements in view `v` for which the `pred` is true.


### Parameters and Requirements
- `execspace`, the InputIterators `first`, `last` and `view` similar to [#Kokkos::Experimental::find_if](#kokkosexperimentalfind_if)
- `label`:
  - 1: The default string is "Kokkos::count_if_iterator_api_default"
  - 3: The default string is "Kokkos::count_if_view_api_default"
- `pred` is similar to [Kokkos::Experimental::equal](#kokkosexperimentalequal)

## Kokkos::Experimental::count

```cpp
template <class ExecutionSpace, class IteratorType, class T>
typename IteratorType::difference_type count(const ExecutionSpace& ex,
                                             IteratorType first,
                                             IteratorType last,                                                (1)
                                             const T& value);
                                             
template <class ExecutionSpace, class IteratorType, class T>
typename IteratorType::difference_type count(const std::string& label,
                                             const ExecutionSpace& ex,
                                             IteratorType first,
                                             IteratorType last,                                                (2)
                                             const T& value);
                                             
template <class ExecutionSpace, class DataType, class... Properties, class T>
auto count(const ExecutionSpace& ex,                                                                            (3)
           const ::Kokkos::View<DataType, Properties...>& view, const T& value);
           
template <class ExecutionSpace, class DataType, class... Properties, class T>
auto count(const std::string& label, const ExecutionSpace& ex,                                                  (4)
           const ::Kokkos::View<DataType, Properties...>& view, const T& value);
           
```

### Description
Returns the number of elements in the range [first,last) that are equal to `value` in (1,2).
For (3,4) returns the number of elements in `v` that are equal to `value`.

### Parameters and Requirements
- `execspace`, `first`, `last` and `view` similar to [#Kokkos::Expeimental::count_if](#kokkosexperimentalcount_if).
- `label`:
    - 1: The default string is "Kokkos::count_iterator_api_default".
    - 3: The default string is "Kokkos::count_view_api_default".
- `pred` - similar to [#Kokkos::Expeimental::count_if](#kokkosexperimentalcount_if)


## Kokkos::Experimental::all_off

```cpp
template <class ExecutionSpace, class InputIterator, class Predicate>
bool all_of(const ExecutionSpace& ex, InputIterator first, InputIterator last,                                       (1)
            Predicate predicate);
            
template <class ExecutionSpace, class InputIterator, class Predicate>
bool all_of(const std::string& label, const ExecutionSpace& ex,                                                      (2)
            InputIterator first, InputIterator last, Predicate predicate); 
            
template <class ExecutionSpace, class DataType, class... Properties,
          class Predicate>
bool all_of(const ExecutionSpace& ex,
            const ::Kokkos::View<DataType, Properties...>& view,                                                     (3)
            Predicate predicate); 

template <class ExecutionSpace, class DataType, class... Properties,
          class Predicate>
bool all_of(const std::string& label, const ExecutionSpace& ex,
            const ::Kokkos::View<DataType, Properties...>& view,                                                     (4)
            Predicate predicate);
```
### Description
Returns `true` if all the elements in the range [first,last) return `true` for unary predicate `pred` in (1,2).
Returns `true` if all the elements in `view` return `true` for unary predicate `pred` in (3,4).

### Parameters and Requirements
- `execspace`, `first`, `last` and `view` similar to [#Kokkos::Expeimental::count](#kokkosexperimentalcount).
- `label`:
    - 1: The default string is "Kokkos::all_of_iterator_api_default".
    - 3: The default string is "Kokkos::all_of_view_api_default".
- `pred` - similar to [#Kokkos::Experimental::count](#kokkosexperimentalcount)

## Kokkos::Experimental::any_of

```cpp
template <class ExecutionSpace, class InputIterator, class Predicate>
bool any_of(const ExecutionSpace& ex, InputIterator first, InputIterator last,                                       (1)
            Predicate predicate);

template <class ExecutionSpace, class InputIterator, class Predicate>
bool any_of(const std::string& label, const ExecutionSpace& ex,
            InputIterator first, InputIterator last, Predicate predicate);                                           (2)

template <class ExecutionSpace, class DataType, class... Properties,
          class Predicate>
bool any_of(const ExecutionSpace& ex,
            const ::Kokkos::View<DataType, Properties...>& v,                                                        (3)
            Predicate predicate);

template <class ExecutionSpace, class DataType, class... Properties,
          class Predicate>
bool any_of(const std::string& label, const ExecutionSpace& ex,                                                      (4)
            const ::Kokkos::View<DataType, Properties...>& v,
            Predicate predicate);
```

### Description
Returns `true` if at least one element in the range [first,last) returns `true` for unary predicate `pred` in (1,2).
Returns `true` if at least one element in `view` returns `true` for unary predicate `pred` in (3,4).

### Parameters and Requirements
- `execspace`, `first`, `last` and `view` similar to [#Kokkos::Expeimental::all_of](#kokkosexperimentalall_of).
- `label`:
    - 1: The default string is "Kokkos::any_of_iterator_api_default".
    - 3: The default string is "Kokkos::any_of_view_api_default".
- `pred` - similar to [#Kokkos::Experimental::all_of](#kokkosexperimentalall_of)

## Kokkos::Experimental::none_of

```cpp
template <class ExecutionSpace, class IteratorType, class Predicate>
bool none_of(const ExecutionSpace& ex, IteratorType first, IteratorType last,                                        (1)
             Predicate predicate);

template <class ExecutionSpace, class IteratorType, class Predicate>
bool none_of(const std::string& label, const ExecutionSpace& ex,                                                     (2)
             IteratorType first, IteratorType last, Predicate predicate);

template <class ExecutionSpace, class DataType, class... Properties,
          class Predicate>
bool none_of(const ExecutionSpace& ex,
             const ::Kokkos::View<DataType, Properties...>& v,                                                       (3)
             Predicate predicate);

template <class ExecutionSpace, class DataType, class... Properties,
          class Predicate>
bool none_of(const std::string& label, const ExecutionSpace& ex,                                                     (4)
             const ::Kokkos::View<DataType, Properties...>& v,
             Predicate predicate);
```
### Description
Returns `true` if no element in the range [first,last) returns `true` for unary predicate `pred` in (1,2).
Returns `true` if no element in `view` returns `true` for unary predicate `pred` in (3,4).

### Parameters and Requirements
- `execspace`, `first`, `last` and `view` similar to [#Kokkos::Expeimental::all_of](#kokkosexperimentalall_of).
- `label`:
    - 1: The default string is "Kokkos::none_of_iterator_api_default".
    - 3: The default string is "Kokkos::none_of_view_api_default".
- `pred` - similar to [#Kokkos::Experimental::all_of](#kokkosexperimentalall_of)

## Kokkos::Experimental::adjacent_find

```cpp
template <class ExecutionSpace, class IteratorType>
IteratorType adjacent_find(const ExecutionSpace& ex, IteratorType first,                                             (1)
                           IteratorType last);

template <class ExecutionSpace, class IteratorType>
IteratorType adjacent_find(const std::string& label, const ExecutionSpace& ex,                                       (2)
                           IteratorType first, IteratorType last);

template <class ExecutionSpace, class DataType, class... Properties>
auto adjacent_find(const ExecutionSpace& ex,                                                                         (3)
                   const ::Kokkos::View<DataType, Properties...>& view);

template <class ExecutionSpace, class DataType, class... Properties>
auto adjacent_find(const std::string& label, const ExecutionSpace& ex,                                               (4)
                   const ::Kokkos::View<DataType, Properties...>& view);

template <class ExecutionSpace, class IteratorType, class BinaryPredicateType>
IteratorType adjacent_find(const ExecutionSpace& ex, IteratorType first,                                             (5)
                           IteratorType last, BinaryPredicateType pred);

template <class ExecutionSpace, class IteratorType, class BinaryPredicateType>
IteratorType adjacent_find(const std::string& label, const ExecutionSpace& ex,                                       (6)
                           IteratorType first, IteratorType last,
                           BinaryPredicateType pred);

template <class ExecutionSpace, class DataType, class... Properties,
          class BinaryPredicateType>
auto adjacent_find(const ExecutionSpace& ex,
                   const ::Kokkos::View<DataType, Properties...>& view,                                              (7)
                   BinaryPredicateType pred);

template <class ExecutionSpace, class DataType, class... Properties,
          class BinaryPredicateType>
auto adjacent_find(const std::string& label, const ExecutionSpace& ex,                                               (8)
                   const ::Kokkos::View<DataType, Properties...>& view,
                   BinaryPredicateType pred);
```

### Description
Searches the range [first,last) for two consecutive equal elements in (1,2,5,6).
Searches `view`  for two consecutive equal elements in (3,4,7,8).
Returns the first iterator `it` in (1,2), where `*it == *it+1` returns true. 
Returns the first iterator `it` in (5,6), where `pred(*it, *it+1)` returns true. 
Returns the first Kokkos view iterator `it` in (3,4), where `view(it) == view(it+1)` returns true. 
Returns the first Kokkos view iterator `it` in (7,8), where `pred(view(it), view(it+1))` returns true. 

### Parameters and Requirements
- `execspace`, `first`, `last` and `view` similar to [#Kokkos::Expeimental::all_of](#kokkosexperimentalall_of).
- `label`:
    - 1,5: The default string is "Kokkos::adjacent_find_iterator_api_default".
    - 3,7: The default string is "Kokkos::adjacent_find_view_api_default".
- `pred` - similar to [Kokkos::Experimental::equal](#kokkosexperimentalequal)

## Kokkos::Experimental::lexicographical_compare

```cpp
template <class ExecutionSpace, class IteratorType1, class IteratorType2>
bool lexicographical_compare(const ExecutionSpace& ex, IteratorType1 first1,
                             IteratorType1 last1, IteratorType2 first2,                                               (1)
                             IteratorType2 last2);

template <class ExecutionSpace, class IteratorType1, class IteratorType2>
bool lexicographical_compare(const std::string& label, const ExecutionSpace& ex,
                             IteratorType1 first1, IteratorType1 last1,                                               (2)
                             IteratorType2 first2, IteratorType2 last2);

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2>
bool lexicographical_compare(
    const ExecutionSpace& ex,
    const ::Kokkos::View<DataType1, Properties1...>& view1,                                                           (3)
    ::Kokkos::View<DataType2, Properties2...>& view2);

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2>
bool lexicographical_compare(
    const std::string& label, const ExecutionSpace& ex,
    const ::Kokkos::View<DataType1, Properties1...>& view1,                                                           (4)
    ::Kokkos::View<DataType2, Properties2...>& view2);

template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class ComparatorType>
bool lexicographical_compare(const ExecutionSpace& ex, IteratorType1 first1,
                             IteratorType1 last1, IteratorType2 first2,                                               (5)
                             IteratorType2 last2, ComparatorType comp);

template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class ComparatorType>
bool lexicographical_compare(const std::string& label, const ExecutionSpace& ex,
                             IteratorType1 first1, IteratorType1 last1,                                               (6)
                             IteratorType2 first2, IteratorType2 last2,
                             ComparatorType comp);

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class ComparatorType>
bool lexicographical_compare(
    const ExecutionSpace& ex,
    const ::Kokkos::View<DataType1, Properties1...>& view1,
    ::Kokkos::View<DataType2, Properties2...>& view2, ComparatorType comp);                                           (7)


template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class ComparatorType>
bool lexicographical_compare(
    const std::string& label, const ExecutionSpace& ex,
    const ::Kokkos::View<DataType1, Properties1...>& view1,                                                           (8)
    ::Kokkos::View<DataType2, Properties2...>& view2, ComparatorType comp);
```

### Description
Returns `true` for (1,2,5,6) if the first range [first1, last1) is lexicographically less than the second range [first2, last2).  
Retruns `true` for (3,4,7,8) if elements in `view1` are lexicographically less than elements in `view2`.
Elements (1,2,3,4) are compared using the `<` operator.
Elements (5,6,7,8) are compared using `comp`.

### Parameters and Requirements
- `execspace`, `first1`, `last1`, `first2`, `last2`, `view1` and `view2` similar to [#Kokkos::Expeimental::mismatch](#kokkosexperimentalmismatch).
- `label`:
    - 1,5: The default string is "Kokkos::lexicographical_compare_iterator_api_defaul".
    - 3,7: The default string is "Kokkos::lexicographical_compare_view_api_default".
- `pred` - similar to [Kokkos::Experimental::equal](#kokkosexperimentalequal)


## Kokkos::Experimental::search

```cpp
template <class ExecutionSpace, class IteratorType1, class IteratorType2>
IteratorType1 search(const ExecutionSpace& ex, IteratorType1 first,
                     IteratorType1 last, IteratorType2 s_first,                                                       (1)
                     IteratorType2 s_last);

template <class ExecutionSpace, class IteratorType1, class IteratorType2>
IteratorType1 search(const std::string& label, const ExecutionSpace& ex,
                     IteratorType1 first, IteratorType1 last,                                                         (2)
                     IteratorType2 s_first, IteratorType2 s_last);

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2>
auto search(const ExecutionSpace& ex,
            const ::Kokkos::View<DataType1, Properties1...>& view,                                                    (3)
            const ::Kokkos::View<DataType2, Properties2...>& s_view);

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2>
auto search(const std::string& label, const ExecutionSpace& ex,
            const ::Kokkos::View<DataType1, Properties1...>& view,                                                    (4)
            const ::Kokkos::View<DataType2, Properties2...>& s_view);

// overload set 2: binary predicate passed
template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class BinaryPredicateType>
IteratorType1 search(const ExecutionSpace& ex, IteratorType1 first,                                                   (5)
                     IteratorType1 last, IteratorType2 s_first,
                     IteratorType2 s_last, const BinaryPredicateType& pred);

template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class BinaryPredicateType>
IteratorType1 search(const std::string& label, const ExecutionSpace& ex,
                     IteratorType1 first, IteratorType1 last,                                                         (6)
                     IteratorType2 s_first, IteratorType2 s_last,
                     const BinaryPredicateType& pred);

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class BinaryPredicateType>
auto search(const ExecutionSpace& ex,
            const ::Kokkos::View<DataType1, Properties1...>& view,                                                    (7)
            const ::Kokkos::View<DataType2, Properties2...>& s_view,
            const BinaryPredicateType& pred);

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class BinaryPredicateType>
auto search(const std::string& label, const ExecutionSpace& ex,
            const ::Kokkos::View<DataType1, Properties1...>& view,                                                    (8)
            const ::Kokkos::View<DataType2, Properties2...>& s_view,
            const BinaryPredicateType& pred)
```

### Description
Searches for the first occurrence of the sequence of elements [s_first, s_last) in the range [first, last) in (1,2,5,6).
Searches for the first occurrence of the sequence of elements `s_view` in `view` in (3,4,7,8).
Elements in (1,2,3,4) are compared using `==` and elements in (5,6,7,8) are compared using `pred`.

### Parameters and Requirements
- `execspace`, `s_first`, `s_last`, `first`, `last`, `s_view` and `view` similar to [#Kokkos::Expeimental::mismatch](#kokkosexperimentalmismatch).
- `label`:
    - 1,5: The default string is "Kokkos::search_iterator_api_default".
    - 3,7: The default string is "Kokkos::search_view_api_default".
- `pred` - similar to [Kokkos::Experimental::equal](#kokkosexperimentalequal)

## Kokkos::Experimental::search_n

```cpp
template <class ExecutionSpace, class IteratorType, class SizeType,
          class ValueType>
IteratorType search_n(const ExecutionSpace& ex, IteratorType first,
                      IteratorType last, SizeType count,                                                              (1)
                      const ValueType& value);

template <class ExecutionSpace, class IteratorType, class SizeType,
          class ValueType>
IteratorType search_n(const std::string& label, const ExecutionSpace& ex,
                      IteratorType first, IteratorType last, SizeType count,                                          (2)
                      const ValueType& value);

template <class ExecutionSpace, class DataType, class... Properties,
          class SizeType, class ValueType>
auto search_n(const ExecutionSpace& ex,
              const ::Kokkos::View<DataType, Properties...>& view,                                                    (3)
              SizeType count, const ValueType& value);

template <class ExecutionSpace, class DataType, class... Properties,
          class SizeType, class ValueType>
auto search_n(const std::string& label, const ExecutionSpace& ex,
              const ::Kokkos::View<DataType, Properties...>& view,                                                    (4)
              SizeType count, const ValueType& value);

// overload set 2: binary predicate passed
template <class ExecutionSpace, class IteratorType, class SizeType,
          class ValueType, class BinaryPredicateType>
IteratorType search_n(const ExecutionSpace& ex, IteratorType first,
                      IteratorType last, SizeType count, const ValueType& value,                                      (5)
                      const BinaryPredicateType& pred);

template <class ExecutionSpace, class IteratorType, class SizeType,
          class ValueType, class BinaryPredicateType>
IteratorType search_n(const std::string& label, const ExecutionSpace& ex,
                      IteratorType first, IteratorType last, SizeType count,                                          (6)
                      const ValueType& value, const BinaryPredicateType& pred);

template <class ExecutionSpace, class DataType, class... Properties,
          class SizeType, class ValueType, class BinaryPredicateType>
auto search_n(const ExecutionSpace& ex,
              const ::Kokkos::View<DataType, Properties...>& view,                                                    (7)
              SizeType count, const ValueType& value,
              const BinaryPredicateType& pred);

template <class ExecutionSpace, class DataType, class... Properties,
          class SizeType, class ValueType, class BinaryPredicateType>
auto search_n(const std::string& label, const ExecutionSpace& ex,
              const ::Kokkos::View<DataType, Properties...>& view,                                                    (8)
              SizeType count, const ValueType& value,
              const BinaryPredicateType& pred);
```

### Description
Searches the range [first, last) for a range of `count` elements each comparing equal to `value`  (1,2).
Searches the `view` for `count` elements each comparing equal to `value`  (3,4).
Searches the range [first, last) for a range of `count` elements for which the `pred` returns true for `value` in (5,6).
Searches the `view` for a range of `count` elements for which the `pred` returns true for `value` in (7,8).

### Parameters and Requirements
- `execspace`, `first`, `last`, `view` and `count` similar to [#Kokkos::Expeimental::for_each_n](#kokkosexperimentalfor_each_n).
- `label`:
    - 1,5: The default string is "Kokkos::search_n_iterator_api_default".
    - 3,7: The default string is "Kokkos::search_n_view_api_default".
- `pred` - similar to [Kokkos::Experimental::equal](#kokkosexperimentalequal)

## Kokkos::Experimental::find_first_of

```cpp
template <class ExecutionSpace, class IteratorType1, class IteratorType2>
IteratorType1 find_first_of(const ExecutionSpace& ex, IteratorType1 first,
                            IteratorType1 last, IteratorType2 s_first,                                                (1)
                            IteratorType2 s_last);

template <class ExecutionSpace, class IteratorType1, class IteratorType2>
IteratorType1 find_first_of(const std::string& label, const ExecutionSpace& ex,
                            IteratorType1 first, IteratorType1 last,                                                  (2)
                            IteratorType2 s_first, IteratorType2 s_last);

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2>
auto find_first_of(const ExecutionSpace& ex,
                   const ::Kokkos::View<DataType1, Properties1...>& view,                                             (3)
                   const ::Kokkos::View<DataType2, Properties2...>& s_view);

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2>
auto find_first_of(const std::string& label, const ExecutionSpace& ex,
                   const ::Kokkos::View<DataType1, Properties1...>& view,                                             (4)
                   const ::Kokkos::View<DataType2, Properties2...>& s_view);

// overload set 2: binary predicate passed
template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class BinaryPredicateType>
IteratorType1 find_first_of(const ExecutionSpace& ex, IteratorType1 first,
                            IteratorType1 last, IteratorType2 s_first,                                                (5)
                            IteratorType2 s_last,
                            const BinaryPredicateType& pred);

template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class BinaryPredicateType>
IteratorType1 find_first_of(const std::string& label, const ExecutionSpace& ex,
                            IteratorType1 first, IteratorType1 last,                                                  (6)
                            IteratorType2 s_first, IteratorType2 s_last,
                            const BinaryPredicateType& pred);

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class BinaryPredicateType>
auto find_first_of(const ExecutionSpace& ex,
                   const ::Kokkos::View<DataType1, Properties1...>& view,                                             (7)
                   const ::Kokkos::View<DataType2, Properties2...>& s_view,
                   const BinaryPredicateType& pred);

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class BinaryPredicateType>
auto find_first_of(const std::string& label, const ExecutionSpace& ex,
                   const ::Kokkos::View<DataType1, Properties1...>& view,                                             (8)
                   const ::Kokkos::View<DataType2, Properties2...>& s_view,
                   const BinaryPredicateType& pred);
```

### Description
Searches the range [first, last) for any of the elements in the range [s_first, s_last) in (1,2).
Searches `view` for any of the elements in `s_view` in (3,4).
Searches the range [first, last) for any of the elements in the range [s_first, s_last) in (5,6) for which `pred` evaluates to `true`.
Searches `view` for any of the elements in `s_view` in (7,8) for which `pred` evaluates to `true`.

### Parameters and Requirements
- `execspace`, `first`, `last`, `view` and `count` similar to [#Kokkos::Expeimental::for_each_n](#kokkosexperimentalfor_each_n).
- `label`:
    - 1,5: The default string is "Kokkos::find_first_of_iterator_api_default".
    - 3,7: The default string is ""Kokkos::find_first_of_view_api_default".
- `pred` - similar to [Kokkos::Experimental::equal](#kokkosexpeimentalequal)


## Kokkos::Experimental::find_end

```cpp
template <class ExecutionSpace, class IteratorType1, class IteratorType2>
IteratorType1 find_end(const ExecutionSpace& ex, IteratorType1 first,
                       IteratorType1 last, IteratorType2 s_first,                                                    (1)
                       IteratorType2 s_last);

template <class ExecutionSpace, class IteratorType1, class IteratorType2>
IteratorType1 find_end(const std::string& label, const ExecutionSpace& ex,
                       IteratorType1 first, IteratorType1 last,                                                       (2)
                       IteratorType2 s_first, IteratorType2 s_last);

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2>
auto find_end(const ExecutionSpace& ex,
              const ::Kokkos::View<DataType1, Properties1...>& view,                                                  (3)
              const ::Kokkos::View<DataType2, Properties2...>& s_view);

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2>
auto find_end(const std::string& label, const ExecutionSpace& ex,
              const ::Kokkos::View<DataType1, Properties1...>& view,                                                  (4)
              const ::Kokkos::View<DataType2, Properties2...>& s_view);

// overload set 2: binary predicate passed
template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class BinaryPredicateType>
IteratorType1 find_end(const ExecutionSpace& ex, IteratorType1 first,
                       IteratorType1 last, IteratorType2 s_first,                                                     (5)
                       IteratorType2 s_last, const BinaryPredicateType& pred);

template <class ExecutionSpace, class IteratorType1, class IteratorType2,
          class BinaryPredicateType>
IteratorType1 find_end(const std::string& label, const ExecutionSpace& ex,
                       IteratorType1 first, IteratorType1 last,                                                       (6)
                       IteratorType2 s_first, IteratorType2 s_last,
                       const BinaryPredicateType& pred);

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class BinaryPredicateType>
auto find_end(const ExecutionSpace& ex,
              const ::Kokkos::View<DataType1, Properties1...>& view,                                                  (7)
              const ::Kokkos::View<DataType2, Properties2...>& s_view,
              const BinaryPredicateType& pred);

template <class ExecutionSpace, class DataType1, class... Properties1,
          class DataType2, class... Properties2, class BinaryPredicateType>
auto find_end(const std::string& label, const ExecutionSpace& ex,
              const ::Kokkos::View<DataType1, Properties1...>& view,                                                  (8)
              const ::Kokkos::View<DataType2, Properties2...>& s_view,
              const BinaryPredicateType& pred);
```

### Description
Searches for the last occurrence of the sequence [s_first, s_last) in the range [first, last) in (1,2).
Searches for the last occurrence of the `s_view` in `view`  in (3,4).
Searches for the last occurrence of the sequence [s_first, s_last) in the range [first, last) in (5,6) for which `pred` evaluates to true.
Searches for the last occurrence of the `s_view` in `view`  in (7,8) for which `pred` evaluates to true.

### Parameters and Requirements
- `execspace`, `first`, `last`, `view` and `count` similar to [#Kokkos::Expeimental::for_each_n](#kokkosexpeimentalfor_each_n).
- `label`:
    - 1,5: The default string is "Kokkos::search_n_iterator_api_default".
    - 3,7: The default string is "Kokkos::search_n_view_api_default".
- `pred` - similar to [Kokkos::Experimental::equal](#kokkosexpeimentalequal)

### Parameters and Requirements
- `execspace`, `s_first`, `s_last`, `first`, `last`, `s_view` and `view` similar to [#Kokkos::Expeimental::search](#kokkosexpeimentalsearch).
- `label`:
    - 1,5: The default string is "Kokkos::find_end_iterator_api_default".
    - 3,7: The default string is "Kokkos::find_end_view_api_default".
- `pred` - similar to [Kokkos::Experimental::equal](#kokkosexperimentalequal)
