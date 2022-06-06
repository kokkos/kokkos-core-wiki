
# Modifying Sequence

Header File: `Kokkos_Core.hpp`

**All algorithms are currently in the `Kokkos::Experimental` namespace.**

Currently supported (see below the full details):
 - [`fill`](#kokkosexperimentalfill)
 - [`fill_n`](#kokkosexperimentalfill_n)
 - [`replace`](#kokkosexperimentalreplace)
 - [`replace_if`](#kokkosexperimentalreplace_if)
 - [`replace_copy`](#kokkosexperimentalreplace_copy)
 - [`replace_copy_if`](#kokkosexperimentalreplace_copy_if)
 - [`copy`](#kokkosexperimentalcopy)
 - [`copy_n`](#kokkosexperimentalcopy_n)
 - [`copy_backward`](#kokkosexperimentalcopy_backward)
 - [`copy_if`](#kokkosexperimentalcopy_if)
 - [`generate`](#kokkosexperimentalgenerate)
 - [`generate_n`](#kokkosexperimentalgenerate_n)
 - [`transform`](#kokkosexperimentaltransform)
 - [`reverse`](#kokkosexperimentalreverse)
 - [`reverse_copy`](#kokkosexperimentalreverse_copy)
 - [`move`](#kokkosexperimentalmove)
 - [`move_backward`](#kokkosexperimentalmove_backward)
 - [`swap_ranges`](#kokkosexperimentalswap_ranges)
 - [`unique`](#kokkosexperimentalunique)
 - [`unique_copy`](#kokkosexperimentalunique_copy)
 - [`rotate`](#kokkosexperimentalrotate)
 - [`rotate_copy`](#kokkosexperimentalrotate_copy)
 - [`remove`](#kokkosexperimentalremove)
 - [`remove_if`](#kokkosexperimentalremove_if)
 - [`remove_copy`](#kokkosexperimentalremove_copy)
 - [`remove_copy_if`](#kokkosexperimentalremove_copy_if)
 - [`shift_left`](#kokkosexperimentalshift_left)
 - [`shift_right`](#kokkosexperimentalshift_right)


------


## `Kokkos::Experimental::fill`

```cpp
template <class ExecutionSpace, class IteratorType, class T>
void fill(const ExecutionSpace& exespace,                                    (1)
          IteratorType first, IteratorType last,
          const T& value);

template <class ExecutionSpace, class IteratorType, class T>
void fill(const std::string& label, const ExecutionSpace& exespace,          (2)
          IteratorType first, IteratorType last,
          const T& value);

template <class ExecutionSpace, class DataType, class... Properties, class T>
void fill(const ExecutionSpace& exespace,                                    (3)
          const Kokkos::View<DataType, Properties...>& view,
          const T& value);

template <class ExecutionSpace, class DataType, class... Properties, class T>
void fill(const std::string& label, const ExecutionSpace& exespace,          (4)
          const Kokkos::View<DataType, Properties...>& view,
          const T& value);
```

### Description

Copy-assigns `value` to each element in the range `[first, last)` (overloads 1,2)
or in `view` (overloads 3,4).


### Parameters and Requirements

- `exespace`:
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::fill_iterator_api_default"
  - for 3, the default string is: "Kokkos::fill_view_api_default"
- `first, last`:
  - range of elements to assign to
  - must be *random access iterators*, e.g., `Kokkos::Experimental::begin/end`
  - must represent a valid range, i.e., `last >= first` (checked in debug mode)
  - must be accessible from `exespace`
- `view`:
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`
- `value`:
  - value to assign to each element


### Return

None

### Example

```cpp
namespace KE = Kokkos::Experimental;
Kokkos::View<double*> a("a", 13);

KE::fill(Kokkos::DefaultExecutionSpace(), KE::begin(a), KE::end(a), 4.);

// passing the view directly
KE::fill(Kokkos::DefaultExecutionSpace(), a, 22.);

// explicitly set execution space (assuming active)
KE::fill(Kokkos::OpenMP(), KE::begin(a), KE::end(a), 14.);
```



-----


## `Kokkos::Experimental::fill_n`

```cpp
template <class ExecutionSpace, class IteratorType, class SizeType, class T>
IteratorType fill_n(const ExecutionSpace& exespace,                             (1)
                    IteratorType first,
                    SizeType n, const T& value);

template <class ExecutionSpace, class IteratorType, class SizeType, class T>
IteratorType fill_n(const std::string& label, const ExecutionSpace& exespace,   (2)
                    IteratorType first,
                    SizeType n, const T& value);

template <class ExecutionSpace, class DataType, class... Properties, class SizeType, class T>
auto fill_n(const ExecutionSpace& exespace,                                     (3)
            const Kokkos::View<DataType, Properties...>& view,
            SizeType n, const T& value);

template <class ExecutionSpace, class DataType, class... Properties, class SizeType, class T>
auto fill_n(const std::string& label, const ExecutionSpace& exespace,           (4)
            const Kokkos::View<DataType, Properties...>& view,
            SizeType n, const T& value);
```

### Description

Copy-assigns `value` to the first `n` elements in the range starting at `first` (overloads 1,2)
or the first `n` elements in `view` (overloads 3,4).

### Parameters and Requirements

- `exespace`,  `first`, `view`, `value`: same as in [`fill`](#kokkosexperimentalfill)
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::fill_n_iterator_api_default"
  - for 3, the default string is: "Kokkos::fill_n_view_api_default"
- `n`:
  - number of elements to modify (must be non-negative)


### Return

If `n > 0`, returns an iterator to the element *after* the last element assigned.

Otherwise, it returns `first` (for 1,2) or `Kokkos::begin(view)` (for 3,4).


### Example

```cpp
namespace KE = Kokkos::Experimental;
Kokkos::View<double*> a("a", 13);
// do something with a
// ...

const double newValue{4};
KE::fill_n(Kokkos::DefaultExecutionSpace(), KE::begin(a), 10, newValue);

// passing the view directly
KE::fill_n(Kokkos::DefaultExecutionSpace(), a, 10, newValue);

// explicitly set execution space (assuming active)
KE::fill_n(Kokkos::OpenMP(), KE::begin(a), 10, newValue);
```


-----


## `Kokkos::Experimental::replace`

```cpp
template <class ExecutionSpace, class IteratorType, class T>
void replace(const ExecutionSpace& exespace,                                 (1)
             IteratorType first, IteratorType last,
             const T& old_value, const T& new_value);

template <class ExecutionSpace, class IteratorType, class T>
void replace(const std::string& label, const ExecutionSpace& exespace,       (2)
             IteratorType first, IteratorType last,
             const T& old_value, const T& new_value);

template <class ExecutionSpace, class DataType, class... Properties, class T>
void replace(const ExecutionSpace& exespace,                                 (3)
             const Kokkos::View<DataType, Properties...>& view,
             const T& old_value, const T& new_value);

template <class ExecutionSpace, class DataType, class... Properties, class T>
void replace(const std::string& label, const ExecutionSpace& exespace,       (4)
             const Kokkos::View<DataType, Properties...>& view,
             const T& old_value, const T& new_value);
```

### Description

Replaces with `new_value` all elements that are equal to `old_value` in the
range `[first, last)` (overloads 1,2) or in `view` (overloads 3,4).
Equality is checked using `operator==`.

### Parameters and Requirements

- `exespace`:
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::replace_iterator_api_default"
  - for 3, the default string is: "Kokkos::replace_view_api_default"
- `first, last`:
  - range of elements to search in
  - must be *random access iterators*, e.g., `Kokkos::Experimental::begin/end`
  - must represent a valid range, i.e., `last >= first` (this condition is checked in debug mode)
  - must be accessible from `exespace`
- `view`:
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`
- `old_value`, `new_value`:
  - self-explanatory


### Return

None


### Example

```cpp
namespace KE = Kokkos::Experimental;
Kokkos::View<double*> a("a", 13);
// do something with a
// ...

const double oldValue{2};
const double newValue{34};
KE::replace(Kokkos::DefaultExecutionSpace(), KE::begin(a), KE::end(a), oldValue, newValue);

// explicitly set label and execution space (assuming active)
KE::replace("mylabel", Kokkos::OpenMP(), a, oldValue, newValue);
```

-----


## `Kokkos::Experimental::replace_if`

```cpp
template <class ExecutionSpace, class IteratorType, class UnaryPredicateType, class T>
void replace_if(const ExecutionSpace& exespace,                              (1)
                IteratorType first, IteratorType last,
                UnaryPredicateType pred, const T& new_value);

template <class ExecutionSpace, class IteratorType, class UnaryPredicateType, class T>
void replace_if(const std::string& label, const ExecutionSpace& exespace,    (2)
                IteratorType first, IteratorType last,
                UnaryPredicateType pred, const T& new_value);

template <class ExecutionSpace, class DataType, class... Properties, class UnaryPredicateType, class T>
void replace_if(const ExecutionSpace& exespace,                              (3)
                const Kokkos::View<DataType, Properties...>& view,
                UnaryPredicateType pred, const T& new_value);

template <class ExecutionSpace, class DataType, class... Properties, class UnaryPredicateType, class T>
void replace_if(const std::string& label, const ExecutionSpace& exespace,    (4)
                const Kokkos::View<DataType, Properties...>& view,
                UnaryPredicateType pred, const T& new_value);
```

### Description

Replaces with `new_value` all the elements for which `pred` is `true` in
the range `[first, last)` (overloads 1,2) or in `view` (overloads 3,4).

### Parameters and Requirements

- `exespace`, `first`, `last`, `view`, `new_value`: same as in [`replace`](#kokkosexperimentalreplace)
- `label`:
  - for 1, the default string is: "Kokkos::replace_if_iterator_api_default"
  - for 3, the default string is: "Kokkos::replace_if_view_api_default"
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

None

### Example

```cpp
template <class ValueType>
struct IsPositiveFunctor {
  KOKKOS_INLINE_FUNCTION
  bool operator()(const ValueType val) const { return (val > 0); }
};
// ---

namespace KE = Kokkos::Experimental;
Kokkos::View<double*> a("a", 13);
// do something with a
// ...

const double oldValue{2};
const double newValue{34};
KE::replace_if(Kokkos::DefaultExecutionSpace(), KE::begin(a), KE::end(a),
   IsPositiveFunctor<double>(), newValue);

// explicitly set label and execution space (assuming active)
KE::replace_if("mylabel", Kokkos::OpenMP(), a,
   IsPositiveFunctor<double>(), newValue);
```


-----


## `Kokkos::Experimental::replace_copy`

```cpp
template <class ExecutionSpace, class InputIteratorType, class OutputIteratorType, class T>
OutputIteratorType replace_copy(const ExecutionSpace& exespace,               (1)
                                InputIteratorType first_from,
                                InputIteratorType last_from,
                                OutputIteratorType first_to,
                                const T& old_value, const T& new_value);

template <class ExecutionSpace, class InputIteratorType, class OutputIteratorType, class T>
OutputIteratorType replace_copy(const std::string& label,                     (2)
                                const ExecutionSpace& exespace,
                                OutputIteratorType first_to,
                                const T& old_value, const T& new_value);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class T
>
auto replace_copy(const ExecutionSpace& exespace,                             (3)
                  const Kokkos::View<DataType1, Properties1...>& view_from,
                  const Kokkos::View<DataType2, Properties2...>& view_to,
                  const T& old_value, const T& new_value);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class T
>
auto replace_copy(const std::string& label,
                  const ExecutionSpace& exespace,                             (4)
                  const Kokkos::View<DataType1, Properties1...>& view_from,
                  const Kokkos::View<DataType2, Properties2...>& view_to,
                  const T& old_value, const T& new_value);
```

### Description

Copies the elements from range `[first_from, last_from)` to another range
beginning at `first_to` (overloads 1,2) or from `view_from` to `view_to`
(overloads 3,4) replacing with `new_value` all elements that equal `old_value`.
Comparison between elements is done using `operator==`.

### Parameters and Requirements

- `exespace`:
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::replace_copy_iterator_api_default"
  - for 3, the default string is: "Kokkos::replace_copy_view_api_default"
- `first_from, last_from`:
  - range of elements to copy from
  - must be *random access iterators*
  - must represent a valid range, i.e., `last_from >= first_from` (checked in debug mode)
  - must be accessible from `exespace`
- `first_to`:
  - beginning of the range to copy to
  - must be a *random access iterator*
  - must be accessible from `exespace`
- `view_from`, `view_to`:
  - source and destination views
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`
- `old_value`, `new_value`:
  - self-explanatory


### Return

Iterator to the element *after* the last element copied.


----


## `Kokkos::Experimental::replace_copy_if`

```cpp
template <
  class ExecutionSpace,
  class InputIteratorType, class OutputIteratorType,
  class UnaryPredicateType, class T
>
OutputIteratorType replace_copy_if(const ExecutionSpace& exespace,              (1)
                                   InputIteratorType first_from,
                                   InputIteratorType last_from,
                                   OutputIteratorType first_to,
                                   UnaryPredicateType pred, const T& new_value);

template <
  class ExecutionSpace,
  class InputIteratorType,  class OutputIteratorType,
  class UnaryPredicateType, class T
>
OutputIteratorType replace_copy_if(const std::string& label,                    (2)
                                   const ExecutionSpace& exespace,
                                   InputIteratorType first_from,
                                   InputIteratorType last_from,
                                   OutputIteratorType first_to,
                                   UnaryPredicateType pred, const T& new_value);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class UnaryPredicateType, class T
>
auto replace_copy_if(const ExecutionSpace& exespace,                            (3)
                     const Kokkos::View<DataType1, Properties1...>& view_from,
                     const Kokkos::View<DataType2, Properties2...>& view_to,
                     UnaryPredicateType pred, const T& new_value);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class UnaryPredicateType, class T
>
auto replace_copy_if(const std::string& label,                                  (4)
                     const ExecutionSpace& exespace,
                     const Kokkos::View<DataType1, Properties1...>& view_from,
                     const Kokkos::View<DataType2, Properties2...>& view_to,
                     UnaryPredicateType pred, const T& new_value);
```

### Description

Copies the elements from range `[first_from, last_from)` to another range
beginning at `first_to` (overloads 1,2) or from `view_from` to `view_to`
(overloads 3,4) replacing with `new_value` all elements for which `pred` returns `true`.


### Parameters and Requirements

- `exespace`, `first_from`, `last_from`, `first_to`, `view_from`, `view_to`, `new_value`:
  - same as in [`replace_copy`](#kokkosexperimentalreplace_copy)
- `label`:
  - for 1, the default string is: "Kokkos::replace_copy_if_iterator_api_default"
  - for 3, the default string is: "Kokkos::replace_copy_if_view_api_default"
- `pred`:
  - unary predicate which returns `true` for the required element; `pred(v)`
  must be valid to be called from the execution space passed, and convertible to bool for every
  argument `v` of type (possible const) `value_type`, where `value_type`
  is the value type of `InputIteratorType` (for 1,2) or of `view_from` (for 3,4),
  and must not modify `v`.
  - should have the same API as that shown for [`replace_if`](#kokkosexperimentalreplace_if)


### Return

Iterator to the element *after* the last element copied.


----


## `Kokkos::Experimental::copy`

```cpp
template <class ExecutionSpace, class InputIteratorType, class OutputIteratorType>
OutputIteratorType copy(const ExecutionSpace& exespace,                      (1)
                        InputIteratorType first_from,
                        InputIteratorType last_from,
                        OutputIteratorType first_to);

template <class ExecutionSpace, class InputIteratorType, class OutputIteratorType>
OutputIteratorType copy(const std::string& label,                            (2)
                        const ExecutionSpace& exespace,
                        InputIteratorType first_from,
                        InputIteratorType last_from,
                        OutputIteratorType first_to);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2
>
auto copy(const ExecutionSpace& exespace,                                    (3)
          const Kokkos::View<DataType1, Properties1...>& view_from,
          const Kokkos::View<DataType2, Properties2...>& view_to);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2
>
auto copy(const std::string& label, const ExecutionSpace& exespace,          (4)
          const Kokkos::View<DataType1, Properties1...>& view_from,
          const Kokkos::View<DataType2, Properties2...>& view_to);
```

### Description

Copies the elements from range `[first_from, last_from)` to another
range beginning at `first_to` (overloads 1,2) or from
a source view `view_from` to a destination view `view_to` (overloads 3,4).

### Parameters and Requirements

- `exespace`:
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::copy_iterator_api_default"
  - for 3, the default string is: "Kokkos::copy_view_api_default"
- `first_from, last_from`:
  - range of elements to copy from
  - must be *random access iterators*
  - must represent a valid range, i.e., `last_from >= first_from` (checked in debug mode)
  - must be accessible from `exespace`
- `first_to`:
  - beginning of the range to copy to
  - must be a *random access iterator*
  - must be accessible from `exespace`
- `view_from`, `view_to`:
  - source and destination views to copy elements from and to
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`


### Return

Iterator to the destination element *after* the last element copied.


----


## `Kokkos::Experimental::copy_n`

```cpp
template <class ExecutionSpace, class InputIteratorType, class SizeType, class OutputIteratorType>
OutputIteratorType copy_n(const ExecutionSpace& exespace,                    (1)
                          InputIteratorType first_from,
                          SizeType n,
                          OutputIteratorType first_to);

template <class ExecutionSpace, class InputIteratorType, class SizeType, class OutputIteratorType>
OutputIteratorType copy_n(const std::string & label,
                          const ExecutionSpace& exespace,                    (2)
                          InputIteratorType first_from,
                          SizeType n,
                          OutputIteratorType first_to);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class SizeType,
  class DataType2, class... Properties2
>
auto copy_n(const ExecutionSpace& exespace,                                  (3)
            const Kokkos::View<DataType1, Properties1...>& view_from,
            SizeType n,
            const Kokkos::View<DataType2, Properties2...>& view_to);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class SizeType,
  class DataType2, class... Properties2
>
auto copy_n(const std::string& label, const ExecutionSpace& exespace,        (4)
            const Kokkos::View<DataType1, Properties1...>& view_from,
            SizeType n,
            const Kokkos::View<DataType2, Properties2...>& view_to);

```

Copies the first `n` elements starting at `first_from` to
another range starting at `first_to` (overloads 1,2) or the first `n` elements
from `view_from` to `view_to` (overloads 3,4).


### Parameters and Requirements

- `exespace`, `first_from`, `first_to`, `view_from`, `view_to`:
  - same as in [`copy`](#kokkosexperimentalcopy)
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::copy_n_if_iterator_api_default"
  - for 3, the default string is: "Kokkos::copy_n_if_view_api_default"
- `n`:
  - number of elements to copy (must be non-negative)


### Return

If `n>0`, returns an iterator to the destination element *after* the last element copied.

Otherwise, returns `first_to` (for 1,2) or `Kokkos::begin(view_to)` (for 3,4).


----


## `Kokkos::Experimental::copy_backward`

```cpp
template <class ExecutionSpace, class InputIteratorType, class OutputIteratorType>
OutputIteratorType copy_backward(const ExecutionSpace& exespace,                (1)
                                 InputIteratorType first_from,
                                 InputIteratorType last_from,
                                 OutputIteratorType last_to);

template <class ExecutionSpace, class InputIteratorType, class OutputIteratorType>
OutputIteratorType copy_backward(const std::string& label,
                                 const ExecutionSpace& exespace,                (2)
                                 InputIteratorType first_from,
                                 InputIteratorType last_from,
                                 OutputIteratorType last_to);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2
>
auto copy_backward(const ExecutionSpace& exespace,                              (3)
                   const Kokkos::View<DataType1, Properties1...>& view_from,
                   const Kokkos::View<DataType2, Properties2...>& view_to);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2
>
auto copy_backward(const std::string& label, const ExecutionSpace& exespace,    (4)
                   const Kokkos::View<DataType1, Properties1...>& view_from,
                   const Kokkos::View<DataType2, Properties2...>& view_to);
```

### Description

Copies the elements in reverse order from range `[first_from, last_from)` to another
range *ending* at `last_to` (overloads 1,2) or from
a source view `view_from` to a destination view `view_to` (overloads 3,4).
The relative order is preserved.

### Parameters and Requirements

- `exespace`:
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::copy_backward_iterator_api_default"
  - for 3, the default string is: "Kokkos::copy_backward_view_api_default"
- `first_from, last_from`:
  - range of elements to copy from
  - must be *random access iterators*
  - must represent a valid range, i.e., `last_from >= first_from` (checked in debug mode)
  - must be accessible from `exespace`
- `last_to`:
  - iterator past the last element of the range to copy to
  - must be a *random access iterator*
  - must be accessible from `exespace`
- `view_from`, `view_to`:
  - source and destination views to copy elements from and to
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`


### Return

Iterator to the last element copied.


----


## `Kokkos::Experimental::copy_if`

```cpp
template <
  class ExecutionSpace, class InputIteratorType, class OutputIteratorType, class UnaryPredicateType
>
OutputIteratorType copy_if(const ExecutionSpace& exespace,                   (1)
                           InputIteratorType first_from,
                           InputIteratorType last_from,
                           OutputIteratorType first_to,
                           UnaryPredicateType pred);

template <
  class ExecutionSpace, class InputIteratorType, class OutputIteratorType, class UnaryPredicateType
>
OutputIteratorType copy_if(const std::string& label,
                           const ExecutionSpace& exespace,                   (2)
                           InputIteratorType first_from,
                           InputIteratorType last_from,
                           OutputIteratorType first_to,
                           UnaryPredicateType pred);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class UnaryPredicateType
>
auto copy_if(const ExecutionSpace& exespace,                                 (3)
             const Kokkos::View<DataType1, Properties1...>& view_from,
             const Kokkos::View<DataType2, Properties2...>& view_to,
             UnaryPredicateType pred);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class UnaryPredicateType
>
auto copy_if(const std::string& label, const ExecutionSpace& exespace,       (4)
             const Kokkos::View<DataType1, Properties1...>& view_from,
             const Kokkos::View<DataType2, Properties2...>& view_to,
             UnaryPredicateType pred);
```

### Description

Copies the elements for which `pred` returns `true` from range `[first_from, last_from)`
to another range beginning at `first_to` (overloads 1,2) or from `view_from` to `view_to`
(overloads 3,4).


### Parameters and Requirements

- `exespace`, `first_from`, `last_from`, `first_to`, `view_from`, `view_to`:
  - same as in [`copy`](#kokkosexperimentalcopy)
- `label`:
  - for 1, the default string is: "Kokkos::copy_if_iterator_api_default"
  - for 3, the default string is: "Kokkos::copy_if_view_api_default"
- `pred`:
  - unary predicate which returns `true` for the required element; `pred(v)`
  must be valid to be called from the execution space passed, and convertible to bool for every
  argument `v` of type (possible const) `value_type`, where `value_type`
  is the value type of `InputIteratorType` (for 1,2) or of `view_from` (for 3,4),
  and must not modify `v`.
  - should have the same API as the unary predicate in [`replace_if`](#kokkosexperimentalreplace_if)


### Return

Iterator to the destination element *after* the last element copied.


----


## `Kokkos::Experimental::generate`

```cpp
template <class ExecutionSpace, class IteratorType, class GeneratorType>
void generate(const ExecutionSpace& exespace,                                (1)
              IteratorType first, IteratorType last,
              GeneratorType g);

template <class ExecutionSpace, class IteratorType, class GeneratorType>
void generate(const std::string& label, const ExecutionSpace& exespace,      (2)
              IteratorType first, IteratorType last,
              GeneratorType g);

template <class ExecutionSpace, class DataType, class... Properties, class GeneratorType>
void generate(const ExecutionSpace& exespace,                                (3)
              const Kokkos::View<DataType, Properties...>& view,
              GeneratorType g);

template <class ExecutionSpace, class DataType, class... Properties, class GeneratorType>
void generate(const std::string& label, const ExecutionSpace& exespace,      (4)
              const Kokkos::View<DataType, Properties...>& view,
              GeneratorType g);
```

### Description

Assigns the value generated by the functor `g` to each elements in the
range `[first, last)` (overloads 1,2) or in the `view` (overloads 3,4).


### Parameters and Requirements

- `exespace`:
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::generate_iterator_api_default"
  - for 3, the default string is: "Kokkos::generate_view_api_default"
- `first, last`:
  - range of elements to modify
  - must be *random access iterators*
  - must represent a valid range, i.e., `last >= first` (checked in debug mode)
  - must be accessible from `exespace`
- `view`:
  - view to modify
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`
- `g`:
  - functor of the form:
  ```cpp
  struct Generate
  {
      KOKKOS_INLINE_FUNCTION
      return_type operator()() const{ return /* ... */; }
  };
  ```
  where `return_type` must be assignable to `value_type`, with `value_type`
  being the value type of `IteratorType` (for 1,2) or of `view` (for 3,4).


### Return

None


----


## `Kokkos::Experimental::generate_n`

```cpp
template <class ExecutionSpace, class IteratorType, class SizeType, class GeneratorType>
IteratorType generate_n(const ExecutionSpace& exespace,                      (1)
                        IteratorType first, SizeType n,
                        GeneratorType g);

template <class ExecutionSpace, class IteratorType, class SizeType, class GeneratorType>
IteratorType generate_n(const std::string& label,
                        const ExecutionSpace& exespace,                      (2)
                        IteratorType first, SizeType n,
                        GeneratorType g);

template <
  class ExecutionSpace, class DataType, class... Properties, class SizeType, class GeneratorType
>
auto generate_n(const ExecutionSpace& exespace,                              (3)
                const Kokkos::View<DataType, Properties...>& view,
                SizeType n, GeneratorType g);

template <
  class ExecutionSpace, class DataType, class... Properties, class SizeType, class GeneratorType
>
auto generate_n(const std::string& label,                                    (4)
                const ExecutionSpace& exespace,
                const Kokkos::View<DataType, Properties...>& view,
                SizeType n, GeneratorType g);
```

### Description

Assigns the value generated by the functor `g` to the first `n` elements
starting at `first` (overloads 1,2) or the first `n` elements in `view` (overloads 3,4).

### Parameters and Requirements

- `exespace`, `first`, `view`, `g`: same as `generate`
- `label`:
  - for 1, the default string is: "Kokkos::generate_n_iterator_api_default"
  - for 3, the default string is: "Kokkos::generate_n_view_api_default"
- `n`:
  - number of elements to modify (must be non-negative)

### Return

If `n>0`, returns an iterator to the element *after* the last element modified.

Otherwise, returns `first` (for 1,2) or `Kokkos::begin(view)` (for 3,4).



----



## `Kokkos::Experimental::transform`

```cpp
template <class ExecutionSpace, class InputIterator, class OutputIterator, class UnaryOperation>
OutputIterator transform(const ExecutionSpace& exespace,                        (1)
                         InputIterator first_from, InputIterator last_from,
                         OutputIterator first_to,
                         UnaryOperation unary_op);

template <class ExecutionSpace, class InputIterator, class OutputIterator, class UnaryOperation>
OutputIterator transform(const std::string& label,                              (2)
                         const ExecutionSpace& exespace,
                         InputIterator first_from, InputIterator last_from,
                         OutputIterator d_first,
                         UnaryOperation unary_op);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class UnaryOperation
>
auto transform(const ExecutionSpace& exespace,                                  (3)
               const Kokkos::View<DataType1, Properties1...>& source,
               Kokkos::View<DataType2, Properties2...>& dest,
               UnaryOperation unary_op);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class UnaryOperation
>
auto transform(const std::string& label, const ExecutionSpace& exespace,        (4)
               const Kokkos::View<DataType1, Properties1...>& source,
               Kokkos::View<DataType2, Properties2...>& dest,
               UnaryOperation unary_op);

template <
  class ExecutionSpace,
  class InputIterator1, class InputIterator2, class OutputIterator,
  class BinaryOperation
>
OutputIterator transform(const ExecutionSpace& exespace,                        (5)
                         InputIterator1 first_from1, InputIterator1 last_from1,
                         InputIterator2 first_from2, OutputIterator first_to,
                         BinaryOperation binary_op);

template <
  class ExecutionSpace,
  class InputIterator1, class InputIterator2, class OutputIterator,
  class BinaryOperation
>
OutputIterator transform(const std::string& label,                              (6)
                         const ExecutionSpace& exespace,
                         InputIterator1 first_from1, InputIterator1 last_from1,
                         InputIterator2 first_from2, OutputIterator first_to,
                         BinaryOperation binary_op);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class DataType3, class... Properties3,
  class BinaryOperation
>
auto transform(const ExecutionSpace& exespace,                                  (7)
               const Kokkos::View<DataType1, Properties1...>& source1,
               const Kokkos::View<DataType2, Properties2...>& source2,
               Kokkos::View<DataType3, Properties3...>& dest,
               BinaryOperation binary_op);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class DataType3, class... Properties3,
  class BinaryOperation
>
auto transform(const std::string& label, const ExecutionSpace& exespace,        (8)
               const Kokkos::View<DataType1, Properties1...>& source1,
               const Kokkos::View<DataType2, Properties2...>& source2,
               Kokkos::View<DataType3, Properties3...>& dest,
               BinaryOperation binary_op);
```

### Description

- Overloads (1,2): applies the given *unary* operation to all elements in the
range `[first_from, last_from)` stores the result in the range starting at `first_to`

- Overloads (3,4): applies the given *unary* operation to all elements in
the `source` view and stores the result in the `dest` view.

- Overloads (5,6): applies the given *binary* operation to pair of elements
from the ranges `[first_from1, last_from1)` and `[first_from2, last_from2]`
and stores the result in range starting at `first_to`

- Overloads (7,8): applies the given *binary* operation to pair of elements
from the views `source1, source2` and stores the result in `dest` view


### Parameters and Requirements

- `exespace`:
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1,3,5,7, the default string is: "Kokkos::transform_iterator_api_default"
  - for 2,4,6,8, the default string is: "Kokkos::transform_view_api_default"
- `first_from, last_from, first_from1, first_from2`:
  - ranges of elements to tranform
  - must be *random access iterators*
  - must be valid ranges, i.e., `first_from >= last_from`, `first_from1 >= last_from2`
  - must be accessible from `exespace`
- `first_to`:
  - beginning of the range to write to
  - must be a *random access iterator*
  - must be accessible from `exespace`
- `source, source1, source2`:
  - source views to transform
  - must be accessible from `exespace`
- `dest`:
  - destination view to write to
  - must be accessible from `exespace`


### Return

Iterator to the element *after* the last element transformed.


-----


## `Kokkos::Experimental::reverse`

```cpp
template <class ExecutionSpace, class IteratorType, class T>
void reverse(const ExecutionSpace& exespace,                                    (1)
             IteratorType first, IteratorType last);

template <class ExecutionSpace, class IteratorType, class T>
void reverse(const std::string& label, const ExecutionSpace& exespace,          (2)
             IteratorType first, IteratorType last);

template <class ExecutionSpace, class DataType, class... Properties, class T>
void reverse(const ExecutionSpace& exespace,                                    (3)
             const Kokkos::View<DataType, Properties...>& view);

template <class ExecutionSpace, class DataType, class... Properties, class T>
void reverse(const std::string& label, const ExecutionSpace& exespace,          (4)
             const Kokkos::View<DataType, Properties...>& view);
```

### Description

Reverses ther order of the elements in the range `[first, last)` (overloads 1,2)
or in `view` (overloads 3,4).


### Parameters and Requirements

- `exespace`:
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::reverse_iterator_api_default"
  - for 3, the default string is: "Kokkos::reverse_view_api_default"
- `first, last`:
  - range of elements to reverse
  - must be *random access iterators*
  - must represent a valid range, i.e., `last >= first` (this condition is checked in debug mode)
  - must be accessible from `exespace`
- `view`:
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`


### Return

None


-----


## `Kokkos::Experimental::reverse_copy`

```cpp
template <class ExecutionSpace, class InputIterator, class OutputIterator>
OutputIterator reverse_copy(const ExecutionSpace& exespace,                  (1)
                            InputIterator first_from,
                            InputIterator last_from,
                            OutputIterator first_to);

template <class ExecutionSpace, class InputIterator, class OutputIterator>
OutputIterator reverse_copy(const std::string& label,                        (2)
                            const ExecutionSpace& exespace,
                            InputIterator first_from,
                            InputIterator last_from,
                            OutputIterator first_to);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2
>
auto reverse_copy(const ExecutionSpace& exespace,                            (3)
                  const Kokkos::View<DataType1, Properties1...>& source,
                  Kokkos::View<DataType2, Properties2...>& dest);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2
>
auto reverse_copy(const std::string& label,                                  (4)
                  const ExecutionSpace& exespace,
                  const Kokkos::View<DataType1, Properties1...>& source,
                  Kokkos::View<DataType2, Properties2...>& dest);
```

### Description

- Overloads 1,2: copies the elements from the range `[first_from, last_from)`
and writes them in reverse order to the range beginning at `first_to`

- Overloads 3,4: copies the elements from `source` view and writes
them in reverse order to `dest` view


### Parameters and Requirements

- `exespace`:
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::reverse_copy_iterator_api_default"
  - for 3, the default string is: "Kokkos::reverse_copy_view_api_default"
- `first_from, last_from`:
  - range of elements to copy from
  - must be *random access iterators*
  - must represent a valid range, i.e., `last_from >= first_from` (checked in debug mode)
  - must be accessible from `exespace`
- `source, dest`:
  - views to copy from and write to, must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`


### Return

Iterator to the element *after* the last element copied.



-----


## `Kokkos::Experimental::move`

```cpp
template <class ExecutionSpace, class InputIterator, class OutputIterator>
OutputIterator move(const ExecutionSpace& exespace,                          (1)
                    InputIterator first_from,
                    InputIterator last_from,
                    OutputIterator first_to);

template <class ExecutionSpace, class InputIterator, class OutputIterator>
OutputIterator move(const std::string& label,                                (2)
                    const ExecutionSpace& exespace,
                    InputIterator first_from,
                    InputIterator last_from,
                    OutputIterator first_to);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2
>
auto move(const ExecutionSpace& exespace,                                    (3)
          const Kokkos::View<DataType1, Properties1...>& source,
          Kokkos::View<DataType2, Properties2...>& dest);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2
>
auto move(const std::string& label,                                          (4)
          const ExecutionSpace& exespace,
          const Kokkos::View<DataType1, Properties1...>& source,
          Kokkos::View<DataType2, Properties2...>& dest);
```

### Description

- Overloads 1,2: moves the elements from the range `[first_from, last_from)`
to the range beginning at `first_to`

- Overloads 3,4: moves the elements from `source` view to `dest` view


### Parameters and Requirements

- `exespace`:
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::move_iterator_api_default"
  - for 3, the default string is: "Kokkos::move_view_api_default"
- `first_from, last_from`:
  - range of elements to move
  - must be *random access iterators*
  - must represent a valid range, i.e., `last_from >= first_from` (checked in debug mode)
  - must be accessible from `exespace`
- `source, dest`:
  - views to move from and to, must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`


### Return

Iterator to the element *after* the last element moved.



-----


## `Kokkos::Experimental::move_backward`

```cpp
template <class ExecutionSpace, class InputIterator, class OutputIterator>
OutputIterator move_backward(const ExecutionSpace& exespace,                 (1)
                             InputIterator first_from,
                             InputIterator last_from,
                             OutputIterator first_to);

template <class ExecutionSpace, class InputIterator, class OutputIterator>
OutputIterator move_backward(const std::string& label,                       (2)
                             const ExecutionSpace& exespace,
                             InputIterator first_from,
                             InputIterator last_from,
                             OutputIterator first_to);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2
>
auto move_backward(const ExecutionSpace& exespace,                           (3)
                   const Kokkos::View<DataType1, Properties1...>& source,
                   Kokkos::View<DataType2, Properties2...>& dest);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2
>
auto move_backward(const std::string& label,                                 (4)
                   const ExecutionSpace& exespace,
                   const Kokkos::View<DataType1, Properties1...>& source,
                   Kokkos::View<DataType2, Properties2...>& dest);
```

### Description

- Overloads 1,2: moves the elements from the range `[first_from, last_from)`
in reverse order to the range beginning at `first_to`

- Overloads 3,4: moves the elements from `source` view in reverse order to `dest` view


### Parameters and Requirements

- `exespace`, `first_from`, `last_from`, `source`, `dest`: same as `Kokkos::Experimental::move`
- `label`:
  - for 1, the default string is: "Kokkos::move_backward_iterator_api_default"
  - for 3, the default string is: "Kokkos::move_backward_view_api_default"

### Return

Iterator to the element *after* the last element moved.


-----


## `Kokkos::Experimental::swap_ranges`

```cpp
template <class ExecutionSpace, class Iterator1, class Iterator2>
Iterator2 swap_ranges(const ExecutionSpace& exespace,                        (1)
                      Iterator1 first1, Iterator1 last1,
                      Iterator2 first2);

template <class ExecutionSpace, class Iterator1, class Iterator2>
Iterator2 swap_ranges(const std::string& label,                              (2)
                      const ExecutionSpace& exespace,
                      Iterator1 first1, Iterator1 last1,
                      Iterator2 first2);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2
>
auto swap_ranges(const ExecutionSpace& exespace,                             (3)
                 const Kokkos::View<DataType1, Properties1...>& source,
                 Kokkos::View<DataType2, Properties2...>& dest);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2
>
auto swap_ranges(const std::string& label,                                   (4)
                 const ExecutionSpace& exespace,
                 const Kokkos::View<DataType1, Properties1...>& source,
                 Kokkos::View<DataType2, Properties2...>& dest);
```

### Description

- Overloads 1,2: swaps the elements between the range `[first1, last1)`
and the range beginning at `first2`

- Overloads 3,4: swaps the elements between `source` and `dest` view


### Parameters and Requirements

- `exespace`:
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::swap_ranges_iterator_api_default"
  - for 3, the default string is: "Kokkos::swap_ranges_view_api_default"
- `first1, last1`, `first2`:
  - iterators to ranges to swap from and to
  - must be *random access iterators*
  - must represent a valid range, i.e., `last1 >= first1` (checked in debug mode)
  - must be accessible from `exespace`
- `source, dest`:
  - views to move from and to, must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`

### Return

Iterator to the element *after* the last element swapped.


-----

### `Kokkos::Experimental::unique`

```cpp
template <class ExecutionSpace, class IteratorType>
IteratorType unique(const ExecutionSpace& exespace,                          (1)
                    IteratorType first, IteratorType last);

template <class ExecutionSpace, class IteratorType>
IteratorType unique(const std::string& label,                                (2)
                    const ExecutionSpace& exespace,
                    IteratorType first, IteratorType last);

template <class ExecutionSpace, class DataType, class... Properties>
auto unique(const ExecutionSpace& exespace,                                  (3)
            const Kokkos::View<DataType, Properties...>& view);

template <class ExecutionSpace, class DataType, class... Properties>
auto unique(const std::string& label, const ExecutionSpace& exespace,        (4)
            const Kokkos::View<DataType, Properties...>& view);

template <class ExecutionSpace, class IteratorType, class BinaryPredicate>
IteratorType unique(const ExecutionSpace& exespace,                          (5)
                    IteratorType first, IteratorType last,
                    BinaryPredicate pred);

template <class ExecutionSpace, class IteratorType, class BinaryPredicate>
IteratorType unique(const std::string& label,                                (6)
                    const ExecutionSpace& exespace,
                    IteratorType first, IteratorType last,
                    BinaryPredicate pred);

template <
  class ExecutionSpace,
  class DataType, class... Properties,
  class BinaryPredicate>
auto unique(const ExecutionSpace& exespace,                                  (7)
            const Kokkos::View<DataType, Properties...>& view,
            BinaryPredicate pred);

template <
  class ExecutionSpace,
  class DataType, class... Properties,
  class BinaryPredicate>
auto unique(const std::string& label,                                        (8)
            const ExecutionSpace& exespace,
            const Kokkos::View<DataType, Properties...>& view,
            BinaryPredicate pred);
```

### Description

- Overloads 1,2,5,6: eliminates all except the first element from every
consecutive group of equivalent elements from the range `[first, last)`,
and returns an iterator to the element *after* the new logical end of the range.
Equivalence is checked using `operator==` for (1,2) and the binary predicate `pred` for (5,6).

- Overloads 3,4,7,8: eliminates all except the first element from every
consecutive group of equivalent elements in `view`, and returns an
iterator to the element *after* the new logical end of the range.
Equivalence is checked using `operator==` for (3,4) and the binary predicate `pred` for (7,8).

### Parameters and Requirements

- `exespace`:
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::unique_iterator_api_default"
  - for 3, the default string is: "Kokkos::unique_ranges_view_api_default"
- `first, last`:
  - iterators to range to examine
  - must be *random access iterators*
  - must represent a valid range, i.e., `last >= first` (checked in debug mode)
  - must be accessible from `exespace`
- `view`:
  - view to examine, must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`

### Return

Iterator to the element *after* the new logical end of the range


-----

### `Kokkos::Experimental::unique_copy`

```cpp
template <class ExecutionSpace, class InputIterator, class OutputIterator>
OutputIterator unique_copy(const ExecutionSpace& exespace,                    (1)
                           InputIterator first_from, InputIterator last_from,
                           OutputIterator first_to);

template <class ExecutionSpace, class InputIterator, class OutputIterator>
OutputIterator unique_copy(const std::string& label,                          (2)
                           const ExecutionSpace& exespace,
                           InputIterator first_from, InputIterator last_from,
                           OutputIterator first_to);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2>
auto unique_copy(const ExecutionSpace& exespace,                              (3)
                 const Kokkos::View<DataType1, Properties1...>& source,
                 const Kokkos::View<DataType2, Properties2...>& dest);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2>
auto unique_copy(const std::string& label,                                    (4)
                 const ExecutionSpace& exespace,
                 const Kokkos::View<DataType1, Properties1...>& source,
                 const Kokkos::View<DataType2, Properties2...>& dest);

template <
  class ExecutionSpace,
  class InputIterator, class OutputIterator,
  class BinaryPredicate>
OutputIterator unique_copy(const ExecutionSpace& exespace,                    (5)
                           InputIterator first_from, InputIterator last_from,
                           OutputIterator first_to,
                           BinaryPredicate pred);

template <
  class ExecutionSpace,
  class InputIterator, class OutputIterator,
  class BinaryPredicate>
OutputIterator unique_copy(const std::string& label,                          (6)
                           const ExecutionSpace& exespace,
                           InputIterator first_from, InputIterator last_from,
                           OutputIterator first_to,
                           BinaryPredicate pred);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class BinaryPredicate>
auto unique_copy(const ExecutionSpace& exespace,                              (7)
                 const Kokkos::View<DataType1, Properties1...>& source,
                 const Kokkos::View<DataType2, Properties2...>& dest,
                 BinaryPredicate pred);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class BinaryPredicate>
auto unique_copy(const std::string& label,                                    (8)
                 const ExecutionSpace& exespace,
                 const Kokkos::View<DataType1, Properties1...>& source,
                 const Kokkos::View<DataType2, Properties2...>& dest,
                 BinaryPredicate pred);
}
```

### Description

- Overloads 1,2,5,6: copies the elements from the range `[first_from, last_from)`
to a range starting at `first_to` such that there are no consecutive equal elements.
It returns an iterator to the element *after* the last element copied in the destination.
Equivalence is checked using `operator==` for (1,2) and the binary predicate `pred` for (5,6).

- Overloads 3,4,7,8: copies the elements from the `source` view to the `dest` view
such that there are no consecutive equal elements.
It returns an iterator to the element *after* the last element copied in the destination view.
Equivalence is checked using `operator==` for (3,4) and the binary predicate `pred` for (7,8).

### Parameters and Requirements

- `exespace`:
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::unique_copy_iterator_api_default"
  - for 3, the default string is: "Kokkos::unique_copy_ranges_view_api_default"
- `first_from, last_from`, `first_to`:
  - iterators to source range `{first,last}_from` and destination range `first_to`
  - must be *random access iterators*
  - must represent a valid range, i.e., `last_from >= first_from` (checked in debug mode)
  - must be accessible from `exespace`
- `source`, `dest`:
  - source and destination views, must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`

### Return

Iterator to the element *after* the last element copied in the destination range or view


-----


## `Kokkos::Experimental::rotate`

```cpp
template <class ExecutionSpace, class IteratorType>
IteratorType rotate(const ExecutionSpace& exespace,                            (1)
                    IteratorType first,
                    IteratorType n_first,
                    IteratorType last);

template <class ExecutionSpace, class IteratorType>
IteratorType rotate(const std::string& label, const ExecutionSpace& exespace,  (2)
                    IteratorType first,
                    IteratorType n_first,
                    IteratorType last);

template <class ExecutionSpace, class DataType, class... Properties>
auto rotate(const ExecutionSpace& exespace,                                    (3)
            const ::Kokkos::View<DataType, Properties...>& view,
            std::size_t n_location);

template <class ExecutionSpace, class DataType, class... Properties>
auto rotate(const std::string& label, const ExecutionSpace& exespace,          (4)
            const ::Kokkos::View<DataType, Properties...>& view,
            std::size_t n_location);
```

### Description

Swaps the elements in the range `[first, last)` or in `view` in such a way that
the element `n_first` or `view(n_location)` becomes the first element of the
new range and `n_first - 1` becomes the last element.


### Parameters and Requirements

- `exespace`:
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::rotate_iterator_api_default"
  - for 3, the default string is: "Kokkos::rotate_view_api_default"
- `first, last`:
  - range of elements to modify
  - must be *random access iterators*
  - must represent a valid range, i.e., `last >= first` (checked in debug mode)
  - must be accessible from `exespace`
- `n_first`:
  - iterator to element that should be the first of the rotated range
  - must be a *random access iterator*
  - must be such that `[first, n_first)` and `[n_first, last)` are valid ranges.
  - must be accessible from `exespace`
- `view`:
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`
- `n_location`:
  - integer value identifying the element to rotate about


### Return

- For (1,2), returns the iterator computed as `first + (last - n_first)`

- For (3,4), returns `Kokkos::begin(view) + (Kokkos::end(view) - n_location)`


-----


## `Kokkos::Experimental::rotate_copy`

```cpp
template <class ExecutionSpace, class InputIterator, class OutputIterator>
OutputIterator rotate_copy(const ExecutionSpace& exespace,                   (1)
                           InputIterator first_from,
                           InputIterator n_first,
                           InputIterator last_from,
                           OutputIterator first_to);

template <class ExecutionSpace, class InputIterator, class OutputIterator>
OutputIterator rotate_copy(const std::string& label,                         (2)
                           const ExecutionSpace& exespace,
                           InputIterator first_from,
                           InputIterator n_first,
                           InputIterator last_from,
                           OutputIterator first_to);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2>
auto rotate_copy(const ExecutionSpace& exespace,                             (3)
                 const ::Kokkos::View<DataType1, Properties1...>& source,
                 std::size_t n_location,
                 const ::Kokkos::View<DataType2, Properties2...>& dest);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2>
auto rotate_copy(const std::string& label, const ExecutionSpace& exespace,   (4)
                 const ::Kokkos::View<DataType1, Properties1...>& source,
                 std::size_t n_location,
                 const ::Kokkos::View<DataType2, Properties2...>& dest);
```

### Description

Copies the elements from the range `[first_from, last_from)` to the range
starting at `first_to` or from `view_from` to `view_dest` in such a way that
the element `n_first` or `view(n_location)` becomes the first element of the
new range and `n_first - 1` becomes the last element.

### Parameters and Requirements

- `exespace`:
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::rotate_copy_iterator_api_default"
  - for 3, the default string is: "Kokkos::rotate_copy_view_api_default"
- `first_from, last_from`:
  - range of elements to copy from
  - must be *random access iterators*
  - must represent a valid range, i.e., `last_from >= first_from` (this condition is checked in debug mode)
  - must be accessible from `exespace`
- `first_to`:
  - beginning of the range to copy to
  - must be a *random access iterator*
  - must be accessible from `exespace`
- `n_first`:
  - iterator to element that should be the first of the rotated range
  - must be a *random access iterator*
  - must be such that `[first_from, n_first)` and `[n_first, last_from)` are valid ranges.
  - must be accessible from `exespace`
- `view_from, view_to`:
  - source and destination views
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`
- `n_location`:
  - integer value identifying the element to rotate about

### Return

Iterator to the element *after* the last element copied.



-----


## `Kokkos::Experimental::remove`

```cpp
template <class ExecutionSpace, class Iterator, class ValueType>
Iterator remove(const ExecutionSpace& exespace,                             (1)
                Iterator first, Iterator last,
                const ValueType& value);

template <class ExecutionSpace, class Iterator, class ValueType>
Iterator remove(const std::string& label,                                   (2)
                const ExecutionSpace& exespace,
                Iterator first, Iterator last,
                const ValueType& value);

template <
  class ExecutionSpace,
  class DataType, class... Properties,
  class ValueType>
auto remove(const ExecutionSpace& exespace,                                 (3)
            const ::Kokkos::View<DataType, Properties...>& view,
            const ValueType& value);

template <
  class ExecutionSpace,
  class DataType, class... Properties,
  class ValueType>
auto remove(const std::string& label,                                       (4)
            const ExecutionSpace& exespace,
            const ::Kokkos::View<DataType, Properties...>& view,
            const ValueType& value);
```

### Description

Removes all elements equal to `value` by shifting via move assignment
the elements in the range `[first, last)` (1,2) or in `view` (3,4)
such that the elements not to be removed
appear in the beginning of the range (1,2) or in the beginning of `view` (3,4).
Relative order of the elements that remain is preserved
and the physical size of the container is unchanged.

### Parameters and Requirements

- `exespace`:
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::remove_iterator_api_default"
  - for 3, the default string is: "Kokkos::remove_view_api_default"
- `first, last`:
  - range of elements to modify
  - must be *random access iterators*
  - must represent a valid range, i.e., `last >= first` (checked in debug mode)
  - must be accessible from `exespace`
- `view`:
  - view of elements to modify
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`
- `value`:
  - target value to remove

### Return

Iterator to the element *after* the new logical end.


-----


## `Kokkos::Experimental::remove_if`

```cpp
template <class ExecutionSpace, class Iterator, class UnaryPredicateType>
Iterator remove_if(const ExecutionSpace& exespace,                           (1)
                   Iterator first, Iterator last,
                   UnaryPredicateType pred);

template <class ExecutionSpace, class Iterator, class UnaryPredicateType>
Iterator remove_if(const std::string& label,                                 (2)
                   const ExecutionSpace& exespace,
                   Iterator first, Iterator last,
                   UnaryPredicateType pred);

template <
  class ExecutionSpace,
  class DataType, class... Properties,
  class UnaryPredicateType>
auto remove_if(const ExecutionSpace& exespace,                               (3)
               const ::Kokkos::View<DataType, Properties...>& view,
               const UnaryPredicateType& pred);

template <
  class ExecutionSpace,
  class DataType, class... Properties,
  class UnaryPredicateType>
auto remove_if(const std::string& label,                                     (4)
         const ExecutionSpace& exespace,
         const ::Kokkos::View<DataType, Properties...>& view,
         const UnaryPredicateType& pred);
```

### Description

Removes all elements for which `pred` returns `true`, by shifting via move assignment
the elements in the range `[first, last)` (1,2) or in `view` (3,4)
such that the elements not to be removed
appear in the beginning of the range (1,2) or in the beginning of `view` (3,4).
Relative order of the elements that remain is preserved
and the physical size of the container is unchanged.


### Parameters and Requirements

- `exespace`, `first`, `last`, `view`: same as in [`remove`](#kokkosexperimentalremove)
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::remove_if_iterator_api_default"
  - for 3, the default string is: "Kokkos::remove_if_view_api_default"
- `pred`:
  - *unary* predicate which returns `true` for the required element; `pred(v)`
  must be valid to be called from the execution space passed, and convertible to bool for every
  argument `v` of type (possible const) `value_type`, where `value_type`
  is the value type of `IteratorType` (for 1,2) or of `view` (for 3,4),
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

Iterator to the element *after* the new logical end.


-----


## `Kokkos::Experimental::remove_copy`

```cpp
template <
  class ExecutionSpace,
  class InputIterator, class OutputIterator,
  class ValueType>
OutputIterator remove_copy(const ExecutionSpace& exespace,                   (1)
                           InputIterator first_from,
                           InputIterator last_from,
                           OutputIterator first_to,
                           const ValueType& value);

template <
  class ExecutionSpace,
  class InputIterator, class OutputIterator,
  class ValueType>
OutputIterator remove_copy(const std::string& label,                         (2)
                           const ExecutionSpace& exespace,
                           InputIterator first_from,
                           InputIterator last_from,
                           OutputIterator first_to,
                           const ValueType& value);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class ValueType>
auto remove_copy(const ExecutionSpace& exespace,                             (3)
                 const ::Kokkos::View<DataType1, Properties1...>& view_from,
                 const ::Kokkos::View<DataType2, Properties2...>& view_dest,
                 const ValueType& value);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class ValueType>
auto remove_copy(const std::string& label,                                   (4)
                 const ExecutionSpace& exespace,
                 const ::Kokkos::View<DataType1, Properties1...>& view_from,
                 const ::Kokkos::View<DataType2, Properties2...>& view_dest,
                 const ValueType& value);
```

### Description

Copies the elements from ther range `[first_from, last_from)` to a new
range starting at `first_to` or from `view_from` to `view_dest` omitting
those that are equal to `value`.


### Parameters and Requirements

- `exespace`:
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::remove_copy_iterator_api_default"
  - for 3, the default string is: "Kokkos::remove_copy_view_api_default"
- `first_from, last_from`:
  - range of elements to copy from
  - must be *random access iterators*
  - must represent a valid range, i.e., `last_from >= first_from` (checked in debug mode)
  - must be accessible from `exespace`
- `first_to`:
  - beginning of the range to copy to
  - must be *random access iterators*
  - must be accessible from `exespace`
- `view_from`, `view_dest`:
  - source and destination views
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`
- `value`:
  - target value to omit

### Return

Iterator to the element after the last element copied.


------


## `Kokkos::Experimental::remove_copy_if`

```cpp
template <
  class ExecutionSpace,
  class InputIterator, class OutputIterator,
  class UnaryPredicateType>
OutputIterator remove_copy_if(const ExecutionSpace& exespace,                   (1)
                              InputIterator first_from,
                              InputIterator last_from,
                              OutputIterator first_to,
                              UnaryPredicateType pred);

template <
  class ExecutionSpace,
  class InputIterator, class OutputIterator,
  class UnaryPredicateType>
OutputIterator remove_copy_if(const std::string& label,                         (2)
                           const ExecutionSpace& exespace,
                           InputIterator first_from,
                           InputIterator last_from,
                           OutputIterator first_to,
                           UnaryPredicateType pred);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class UnaryPredicateType>
auto remove_copy_if(const ExecutionSpace& exespace,                             (3)
                    const ::Kokkos::View<DataType1, Properties1...>& view_from,
                    const ::Kokkos::View<DataType2, Properties2...>& view_dest,
                    UnaryPredicateType pred);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class UnaryPredicateType>
auto remove_copy_if(const std::string& label,                                   (4)
                    const ExecutionSpace& exespace,
                    const ::Kokkos::View<DataType1, Properties1...>& view_from,
                    const ::Kokkos::View<DataType2, Properties2...>& view_dest,
                    UnaryPredicateType pred);
```

### Description

Copies the elements from the range `[first_from, last_from)` to a new
range starting at `first_to` or from `view_from` to `view_dest` omitting
those for which `pred` returns `true`.


### Parameters and Requirements

- `exespace`, `first_from, last_from`, `first_to`, `view_from`, `view_dest`: same as in [`remove_copy`](#kokkosexperimentalremove_copy)
- `label`:
  - for 1, the default string is: "Kokkos::remove_copy_if_iterator_api_default"
  - for 3, the default string is: "Kokkos::remove_copy_if_view_api_default"
- `pred`:
  - *unary* predicate which returns `true` for the required element; `pred(v)`
  must be valid to be called from the execution space passed, and convertible to bool for every
  argument `v` of type (possible const) `value_type`, where `value_type`
  is the value type of `InputIteratorType` (for 1,2) or of `view_from` (for 3,4),
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

Iterator to the element after the last element copied.


------


## `Kokkos::Experimental::shift_left`

```cpp
template <class ExecutionSpace, class IteratorType>
IteratorType shift_left(const ExecutionSpace& exespace,                 (1)
                        IteratorType first,
                        IteratorType last,
                        typename IteratorType::difference_type n);

template <class ExecutionSpace, class IteratorType>
IteratorType shift_left(const std::string& label,                       (2)
                        const ExecutionSpace& exespace,
                        IteratorType first, IteratorType last,
                        typename IteratorType::difference_type n);

template <class ExecutionSpace, class DataType, class... Properties>
auto shift_left(const ExecutionSpace& exespace,                         (3)
                const ::Kokkos::View<DataType, Properties...>& view,
                typename decltype(begin(view))::difference_type n);

template <class ExecutionSpace, class DataType, class... Properties>
auto shift_left(const std::string& label,                               (4)
                const ExecutionSpace& exespace,
                const ::Kokkos::View<DataType, Properties...>& view,
                typename decltype(begin(view))::difference_type n);
```

### Description

Shifts the elements in the range `[first, last)` or in `view`
by `n` positions towards the *beginning*.

### Parameters and Requirements

- `exespace`:
  - execution space
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::shift_left_iterator_api_default"
  - for 3, the default string is: "Kokkos::shift_left_view_api_default"
- `first, last`:
  - range of elements to shift
  - must be *random access iterators*
  - must represent a valid range, i.e., `last >= first` (checked in debug mode)
  - must be accessible from `exespace`
- `view_from`:
  - view to modify
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`
- `n`:
  - the number of positions to shift
  - must be non-negative

### Return

The end of the resulting range.
If `n` is less than `last - first`, returns `first + (last - first - n)`.
Otherwise, returns `first`.


------


## `Kokkos::Experimental::shift_right`

```cpp
template <class ExecutionSpace, class IteratorType>
IteratorType shift_right(const ExecutionSpace& exespace,                  (1)
                         IteratorType first,
                         IteratorType last,
                         typename IteratorType::difference_type n);

template <class ExecutionSpace, class IteratorType>
IteratorType shift_right(const std::string& label,                        (2)
                         const ExecutionSpace& exespace,
                         IteratorType first, IteratorType last,
                         typename IteratorType::difference_type n);

template <class ExecutionSpace, class DataType, class... Properties>
auto shift_right(const ExecutionSpace& exespace,                          (3)
                 const ::Kokkos::View<DataType, Properties...>& view,
                 typename decltype(begin(view))::difference_type n);

template <class ExecutionSpace, class DataType, class... Properties>
auto shift_right(const std::string& label,                                (4)
                 const ExecutionSpace& exespace,
                 const ::Kokkos::View<DataType, Properties...>& view,
                 typename decltype(begin(view))::difference_type n);
```

### Description

Shifts the elements in the range `[first, last)` or in `view`
by `n` positions towards the end of the range or the view.

### Parameters and Requirements

- `exespace`, `first`, `last`, `view`: same as in [`shift_left`](#kokkosexperimentalshift_left)
- `label`:
  - for 1, the default string is: "Kokkos::shift_right_iterator_api_default"
  - for 3, the default string is: "Kokkos::shift_right_view_api_default"
- `n`:
  - the number of positions to shift
  - must be non-negative

### Return

The beginning of the resulting range. If `n` is less than `last - first`,
returns `first + n`. Otherwise, returns `last`.
