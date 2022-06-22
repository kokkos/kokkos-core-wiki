# Numeric

Header File: `Kokkos_Core.hpp`

**All algorithms are currently in the `Kokkos::Experimental` namespace.**

Currently supported (see below the full details):
 - [`reduce`](kokkosexperimentalreduce)
 - [`transform_reduce`](kokkosexperimentaltransform_reduce)
 - [`exclusive_scan`](kokkosexperimentalexclusive_scan)
 - [`transform_exclusive_scan`](kokkosexperimentaltransform_exclusive_scan)
 - [`inclusive_scan`](kokkosexperimentalinclusive_scan)
 - [`transform_inclusive_scan`](kokkosexperimentaltransform_inclusive_scan)
 - [`adjacent_difference`](kokkosexperimentaladjant_difference)

------

(kokkosexperimentalreduce)=
## `Kokkos::Experimental::reduce`

```c++
//
// overload set A
//
template <class ExecutionSpace, class IteratorType>
typename IteratorType::value_type reduce(const ExecutionSpace& exespace,        (1)
                                         IteratorType first,
                                         IteratorType last);

template <class ExecutionSpace, class IteratorType>
typename IteratorType::value_type reduce(const std::string& label,              (2)
                                         const ExecutionSpace& exespace,
                                         IteratorType first,
                                         IteratorType last);

template <class ExecutionSpace, class DataType, class... Properties>
auto reduce(const ExecutionSpace& exespace,                                     (3)
            const ::Kokkos::View<DataType, Properties...>& view);

template <class ExecutionSpace, class DataType, class... Properties>
auto reduce(const std::string& label, const ExecutionSpace& exespace,           (4)
            const ::Kokkos::View<DataType, Properties...>& view);

//
// overload set B
//
template <class ExecutionSpace, class IteratorType, class ValueType>
ValueType reduce(const ExecutionSpace& exespace,                                (5)
                 IteratorType first, IteratorType last,
                 ValueType init_reduction_value);

template <class ExecutionSpace, class IteratorType, class ValueType>
ValueType reduce(const std::string& label, const ExecutionSpace& exespace,      (6)
                 IteratorType first, IteratorType last,
                 ValueType init_reduction_value);

template <class ExecutionSpace, class DataType, class... Properties, class ValueType>
ValueType reduce(const ExecutionSpace& exespace,                                (7)
                 const ::Kokkos::View<DataType, Properties...>& view,
                 ValueType init_reduction_value);

template <class ExecutionSpace, class DataType, class... Properties, class ValueType>
ValueType reduce(const std::string& label,                                      (8)
                 const ExecutionSpace& exespace,
                 const ::Kokkos::View<DataType, Properties...>& view,
                 ValueType init_reduction_value);

//
// overload set C
//
template <
  class ExecutionSpace, class IteratorType, class ValueType,
  class BinaryOp>
ValueType reduce(const ExecutionSpace& exespace,                                (9)
                 IteratorType first, IteratorType last,
                 ValueType init_reduction_value,
                 BinaryOp joiner);

template <
  class ExecutionSpace, class IteratorType, class ValueType,
  class BinaryOp>
ValueType reduce(const std::string& label, const ExecutionSpace& exespace,      (10)
                 IteratorType first, IteratorType last,
                 ValueType init_reduction_value,
                 BinaryOp joiner);

template <
  class ExecutionSpace, class DataType, class... Properties,
  class ValueType, class BinaryOp>
ValueType reduce(const ExecutionSpace& exespace,                                (11)
                 const ::Kokkos::View<DataType, Properties...>& view,
                 ValueType init_reduction_value,
                 BinaryOp joiner);

template <
  class ExecutionSpace, class DataType, class... Properties,
  class ValueType, class BinaryOp>
ValueType reduce(const std::string& label, const ExecutionSpace& exespace,      (12)
                 const ::Kokkos::View<DataType, Properties...>& view,
                 ValueType init_reduction_value,
                 BinaryOp joiner);
```

### Description

- Overload set A (1,2,3,4): performs a reduction of the elements
  in the range `[first, last)` (1,2) or in `view` (3,4).

- Overload set B (5,6,7,8): performs a reduction of the elements
  in the range `[first, last)` (5,6) or in `view` (7,8) accounting for
  the initial value `init_reduction_value`.

- Overload set C (9,10,11,12): performs a reduction of the elements
  in the range `[first, last)` (9,10)
  or in `view` (11,12) accounting for the initial value `init_reduction_value`
  using the functor  `joiner` to join operands during the reduction operation.


### Parameters and Requirements

- `exespace`:
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1,5,9 the default string is: "Kokkos::reduce_iterator_api_default"
  - for 3,7,11 the default string is: "Kokkos::reduce_view_api_default"
- `first`, `last`:
  - range of elements to reduce over
  - must be *random access iterators*
  - must represent a valid range, i.e., `last >= first` (checked in debug mode)
  - must be accessible from `exespace`
- `view`:
  - view to reduce
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`
- `init_reduction_value`:
  - initial reduction value to use
- `joiner`:
  - *binary* functor performing the desired operation to join two elements.
  Must be valid to be called from the execution space passed, and callable with
  two arguments `a,b` of type (possible const) `ValueType`, and must not modify `a,b`.
  - Must conform to:
  ```c++
  struct JoinFunctor {
	KOKKOS_FUNCTION
	constexpr ValueType operator()(const ValueType& a, const ValueType& b) const {
	  return /* ... */
	}

	KOKKOS_FUNCTION
	constexpr ValueType operator()(const volatile ValueType& a,
								   const volatile ValueType& b) const {
	  return /* ... */
	}
  };
  ```
  - The behavior is non-deterministic if the `joiner` operation
  is not associative or not commutative.


### Return

The reduction result.

------

(kokkosexperimentaltransform_reduce)=
## `Kokkos::Experimental::transform_reduce`

```c++
//
// overload set A
//
template <
  class ExecutionSpace, class IteratorType1,
  class IteratorType2, class ValueType>
ValueType transform_reduce(const ExecutionSpace& exespace,                      (1)
                           IteratorType1 first1, IteratorType1 last1,
                           IteratorType2 first2,
                           ValueType init_reduction_value);

template <
  class ExecutionSpace, class IteratorType1,
  class IteratorType2, class ValueType>
ValueType transform_reduce(const std::string& label,                            (2)
                           const ExecutionSpace& exespace,
                           IteratorType1 first1, IteratorType1 last1,
                           IteratorType2 first2,
                           ValueType init_reduction_value);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class ValueType>
ValueType transform_reduce(const ExecutionSpace& exespace,                      (3)
                           const ::Kokkos::View<DataType1, Properties1...>& first_view,
                           const ::Kokkos::View<DataType2, Properties2...>& second_view,
                           ValueType init_reduction_value);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class ValueType>
ValueType transform_reduce(const std::string& label,                            (4)
                           const ExecutionSpace& exespace,
                           const ::Kokkos::View<DataType1, Properties1...>& first_view,
                           const ::Kokkos::View<DataType2, Properties2...>& second_view,
                           ValueType init_reduction_value);

//
// overload set B
//
template <
  class ExecutionSpace,
  class IteratorType1, class IteratorType2,
  class ValueType,
  class BinaryJoinerType, class BinaryTransform>
ValueType transform_reduce(const ExecutionSpace& exespace,                      (5)
                           IteratorType1 first1,
                           IteratorType1 last1, IteratorType2 first2,
                           ValueType init_reduction_value,
                           BinaryJoinerType joiner,
                           BinaryTransform binary_transformer);

template <
  class ExecutionSpace,
  class IteratorType1, class IteratorType2,
  class ValueType,
  class BinaryJoinerType, class BinaryTransform>
ValueType transform_reduce(const std::string& label,                            (6)
                           const ExecutionSpace& exespace,
                           IteratorType1 first1, IteratorType1 last1,
                           IteratorType2 first2,
                           ValueType init_reduction_value,
                           BinaryJoinerType joiner,
                           BinaryTransform binary_transformer);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class ValueType,
  class BinaryJoinerType, class BinaryTransform>
ValueType transform_reduce(const ExecutionSpace& exespace,                      (7)
                           const ::Kokkos::View<DataType1, Properties1...>& first_view,
                           const ::Kokkos::View<DataType2, Properties2...>& second_view,
                           ValueType init_reduction_value,
                           BinaryJoinerType joiner,
                           BinaryTransform binary_transformer);

template <
   class ExecutionSpace,
   class DataType1, class... Properties1,
   class DataType2, class... Properties2,
   class ValueType,
   class BinaryJoinerType, class BinaryTransform>
ValueType transform_reduce(const std::string& label,                            (8)
                           const ExecutionSpace& exespace,
                           const ::Kokkos::View<DataType1, Properties1...>& first_view,
                           const ::Kokkos::View<DataType2, Properties2...>& second_view,
                           ValueType init_reduction_value,
                           BinaryJoinerType joiner,
                           BinaryTransform binary_transformer);

//
// overload set C
//
template <
  class ExecutionSpace,
  class IteratorType, class ValueType,
  class BinaryJoinerType, class UnaryTransform>
ValueType transform_reduce(const ExecutionSpace& exespace,                      (9)
                           IteratorType first1, IteratorType last1,
                           ValueType init_reduction_value,
                           BinaryJoinerType joiner,
                           UnaryTransform unary_transformer);

template <
  class ExecutionSpace,
  class IteratorType, class ValueType,
  class BinaryJoinerType, class UnaryTransform>
ValueType transform_reduce(const std::string& label,
                           const ExecutionSpace& exespace,                      (10)
                           IteratorType first1, IteratorType last1,
                           ValueType init_reduction_value,
                           BinaryJoinerType joiner,
                           UnaryTransform unary_transformer);

template <
  class ExecutionSpace,
  class DataType, class... Properties, class ValueType,
  class BinaryJoinerType, class UnaryTransform>
ValueType transform_reduce(const ExecutionSpace& exespace,                      (11)
                           const ::Kokkos::View<DataType, Properties...>& first_view,
                           ValueType init_reduction_value,
                           BinaryJoinerType joiner,
                           UnaryTransform unary_transformer);

template <
  class ExecutionSpace,
  class DataType, class... Properties, class ValueType,
  class BinaryJoinerType, class UnaryTransform>
ValueType transform_reduce(const std::string& label,                            (12)
                           const ExecutionSpace& exespace,
                           const ::Kokkos::View<DataType, Properties...>& first_view,
                           ValueType init_reduction_value,
                           BinaryJoinerType joiner,
                           UnaryTransform unary_transformer);
```

### Description

- (1,2): performs the *product* (via `operator*`) between each pair
  of elements in the range `[first1, last1)` and the range starting at `first2`,
  and reduces the results along with the initial value `init_reduction_value`

- (3,4): performs the *product* (via `operator*`) between each pair
  of elements in `first_view` and `second_view`,
  and reduces the results along with the initial value `init_reduction_value`

- (5,6): applies the functor `binary_transformer` to each pair of elements
  in the range `[first1, last1)` and the range starting at `first2`,
  and reduces the results along with the initial value `init_reduction_value`
  with the join operation done via the *binary* functor `joiner`

- (7,8): applies the functor `binary_transformer` to each pair of elements
  in `first_view` and `second_view`,
  and reduces the result along with the initial value `init_reduction_value`
  with the join operation done via the *binary* functor `joiner`

- (9,10): applies the functor `unary_transformer` to each element
  in the range `[first, last)`, and reduces the results along with
  the initial value `init_reduction_value`
  with the join operation done via the *binary* functor `joiner`

- (11,12): applies the functor `unary_transformer` to each element
  in `view`, and reduces the results along with
  the initial value `init_reduction_value`
  with the join operation done via the *binary* functor `joiner`


### Parameters and Requirements

- `exespace`:
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1,5,9 the default string is: "Kokkos::transform_reduce_iterator_api_default"
  - for 3,7,11 the default string is: "Kokkos::transform_reduce_view_api_default"
- `first1`, `last1`, `first2`:
  - ranges of elements to transform and reduce
  - must be *random access iterators*
  - must represent a valid range, i.e., `last1 >= first1` (checked in debug mode)
  - must be accessible from `exespace`
- `first_view`, `second_view`:
  - views to transform and reduce
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`
- `init_reduction_value`:
  - initial reduction value to use
- `joiner`:
  - *binary* functor performing the desired operation to join two elements.
  Must be valid to be called from the execution space passed, and callable with
  two arguments `a,b` of type (possible const) `ValueType`, and must not modify `a,b`.
  - Must conform to:
  ```c++
  struct JoinFunctor {
	KOKKOS_FUNCTION
	constexpr ValueType operator()(const ValueType& a,
                                 const ValueType& b) const {
	  return /* ... */
	}

	KOKKOS_FUNCTION
	constexpr ValueType operator()(const volatile ValueType& a,
								   const volatile ValueType& b) const {
	  return /* ... */
	}
  };
  ```
  - The behavior is non-deterministic if the `joiner` operation
  is not associative or not commutative.

- `binary_transformer`:
  - *binary* functor applied to each pair of elements *before* doing the reduction.
  Must be valid to be called from the execution space passed, and callable with
  two arguments `a,b` of type (possible const) `value_type_a` and `value_type_b`,
  where `value_type_{a,b}` are the value types of `first1` and `first2` (for 1,2,5,6)
  or the value types of `first_view` and `second_view` (for 3,4,7,8), and must not modify `a,b`.
  - Must conform to:
  ```c++
  struct BinaryTransformer {
	KOKKOS_FUNCTION
	constexpr return_type operator()(const value_type_a & a, const value_type_b & b) const {
	  return /* ... */
	}
  };
  ```
  - the `return_type` is such that it can be accepted by the `joiner`

- `unary_transformer`:
  - *unary* functor performing the desired operation to an element.
  Must be valid to be called from the execution space passed, and callable with
  an arguments `v` of type (possible const) `value_type`,
  where `value_type` is the value type of `first1` (for 9,10)
  or the value type of `first_view` (for 11,12), and must not modify `v`.
  - Must conform to:
  ```c++
  struct UnaryTransformer {
	KOKKOS_FUNCTION
	constexpr value_type operator()(const value_type & v) const {
	  return /* ... */
	}
  };
  ```

### Return

The reduction result.

------

(kokkosexperimentalexclusive_scan)=
## `Kokkos::Experimental::exclusive_scan`

```c++
//
// overload set A
//
template <
  class ExecutionSpace, class InputIteratorType,
  class OutputIteratorType, class ValueType>
OutputIteratorType exclusive_scan(const ExecutionSpace& exespace,                (1)
                                  InputIteratorType first,
                                  InputIteratorType last,
                                  OutputIteratorType first_dest,
                                  ValueType init_value);

template <
  class ExecutionSpace, class InputIteratorType,
  class OutputIteratorType, class ValueType>
OutputIteratorType exclusive_scan(const std::string& label,                      (2)
                                  const ExecutionSpace& exespace,
                                  InputIteratorType first,
                                  InputIteratorType last,
                                  OutputIteratorType first_dest,
                                  ValueType init_value);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class ValueType>
auto exclusive_scan(const ExecutionSpace& exespace,                              (3)
                    const ::Kokkos::View<DataType1, Properties1...>& view_from,
                    const ::Kokkos::View<DataType2, Properties2...>& view_dest,
                    ValueType init_value);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class ValueType>
auto exclusive_scan(const std::string& label, const ExecutionSpace& exespace,    (4)
                    const ::Kokkos::View<DataType1, Properties1...>& view_from,
                    const ::Kokkos::View<DataType2, Properties2...>& view_dest,
                    ValueType init_value);

//
// overload set B
//
template <
 class ExecutionSpace, class InputIteratorType,
 class OutputIteratorType, class ValueType, class BinaryOpType
 >
OutputIteratorType exclusive_scan(const ExecutionSpace& exespace,                (5)
                                  InputIteratorType first_from,
                                  InputIteratorType last_from,
                                  OutputIteratorType first_dest,
                                  ValueType init_value, BinaryOpType bin_op);


template <
  class ExecutionSpace, class InputIteratorType,
  class OutputIteratorType, class ValueType, class BinaryOpType
  >
OutputIteratorType exclusive_scan(const std::string& label,                      (6)
                                  const ExecutionSpace& exespace,
                                  InputIteratorType first_from,
                                  InputIteratorType last_from,
                                  OutputIteratorType first_dest,
                                  ValueType init_value, BinaryOpType bin_op);

template <
  class ExecutionSpace, class DataType1, class... Properties1,
  class DataType2, class... Properties2, class ValueType,
  class BinaryOpType>
auto exclusive_scan(const ExecutionSpace& exespace,                              (7)
                    const ::Kokkos::View<DataType1, Properties1...>& view_from,
                    const ::Kokkos::View<DataType2, Properties2...>& view_dest,
                    ValueType init_value, BinaryOpType bin_op);

template <
  class ExecutionSpace, class DataType1, class... Properties1,
  class DataType2, class... Properties2, class ValueType,
  class BinaryOpType>
auto exclusive_scan(const std::string& label, const ExecutionSpace& exespace,    (8)
                    const ::Kokkos::View<DataType1, Properties1...>& view_from,
                    const ::Kokkos::View<DataType2, Properties2...>& view_dest,
                    ValueType init_value, BinaryOpType bin_op);
```

### Description

- 1,2,3,4: computes an exclusive prefix *sum* for the range `[first_from, last_from)` (1,2)
or `view_from` (3,4), using `init` as the initial value, and writes
the results to the range beginning at `first_dest` (1,2) or to `view_dest` (3,4).

- 5,6,7,8: computes an exclusive prefix scan using the binary functor `bin_op`
to combine two elements for the range `[first_from, last_from)` (5,6)
or `view_from` (7,8), using `init` as the initial value, and writes
the results to the range beginning at `first_dest` (5,6) or to `view_dest` (7,8).

Exclusive means that the i-th input element is not included in the i-th sum.

### Parameters and Requirements

- `exespace`:
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1,2 the default string is: "Kokkos::exclusive_scan_iterator_api_default"
  - for 5,6 the default string is: "Kokkos::exclusive_scan_view_api_default"
- `first_from`, `last_from`, `first_dest`:
  - range of elements to read from (`*_from`) and write to (`first_dest`)
  - must be *random access iterators*
  - must represent a valid range, i.e., `last_from >= first_from` (checked in debug mode)
  - must be accessible from `exespace`
- `view_from`, `view_dest`:
  - views to read elements from `view_from` and write to `view_dest`
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`
- `bin_op`:
  - *binary* functor representing the operation to combine pair of elements.
  Must be valid to be called from the execution space passed, and callable with
  two arguments `a,b` of type (possible const) `value_type`, where `value_type`
  is the value type of `InputIteratorType` (for 1,2,5,6) or the value type
  of `view_from` (for 3,4,7,8), and must not modify `a,b`.
  - must conform to:
  ```c++
  struct BinaryOp
  {
     KOKKOS_INLINE_FUNCTION
     return_type operator()(const value_type & a,
	                        const value_type & b) const {
       return /* ... */;
     }

     return_type operator()(const volatile value_type & a,
	                  	    const volatile value_type & b) const {
       return /* ... */;
     }
  };
  ```
  The return type `return_type` must be such that an object of type `OutputIteratorType`
  for (1,2,5,6) or an object of type `value_type` where `value_type` is the
  value type of `view_dest` for (3,4,7,8) can be dereferenced and assigned a value of type `return_type`.

  - the volatile overload is needed for correctness by the current Kokkos
  implementation of prefix scan operations


### Return

Iterator to the element *after* the last element written.

------

(kokkosexperimentaltransform_exclusive_scan)=
## `Kokkos::Experimental::transform_exclusive_scan`

```c++
template <
  class ExecutionSpace, class InputIteratorType,
  class OutputIteratorType, class ValueType,
  class BinaryOpType, class UnaryOpType>
OutputIteratorType transform_exclusive_scan(const ExecutionSpace& exespace,     (1)
                                            InputIteratorType first_from,
                                            InputIteratorType last_from,
                                            OutputIteratorType first_dest,
                                            ValueType init_value,
                                            BinaryOpType binary_op,
                                            UnaryOpType unary_op);

template <
  class ExecutionSpace, class InputIteratorType,
  class OutputIteratorType, class ValueType,
  class BinaryOpType, class UnaryOpType>
OutputIteratorType transform_exclusive_scan(const std::string& label,           (2)
                                            const ExecutionSpace& exespace,
                                            InputIteratorType first_from,
                                            InputIteratorType last_from,
                                            OutputIteratorType first_dest,
                                            ValueType init_value,
                                            BinaryOpType binary_op,
                                            UnaryOpType unary_op);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class ValueType, class BinaryOpType, class UnaryOpType>
auto transform_exclusive_scan(const ExecutionSpace& exespace,                   (3)
                              const ::Kokkos::View<DataType1, Properties1...>& view_from,
                              const ::Kokkos::View<DataType2, Properties2...>& view_dest,
                              ValueType init_value, BinaryOpType binary_op,
                              UnaryOpType unary_op);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class ValueType, class BinaryOpType, class UnaryOpType>
auto transform_exclusive_scan(const std::string& label,                         (4)
                              const ExecutionSpace& exespace,
                              const ::Kokkos::View<DataType1, Properties1...>& view_from,
                              const ::Kokkos::View<DataType2, Properties2...>& view_dest,
                              ValueType init_value, BinaryOpType binary_op,
                              UnaryOpType unary_op);
```

### Description

- 1,2: transforms each element in the range `[first_from, last_from)`
with `unary_op`, then computes an exclusive prefix scan operation using `binary_op`
over the resulting range, with `init` as the initial value, and writes
the results to the range beginning at `first_dest`.
"exclusive" means that the i-th input element is not included in the i-th sum

- 3,4: same as (1,2) except that the elements are read from `view_from`
and written to `view_dest`

### Parameters and Requirements

- `exespace`, `first_from`, `first_last`, `first_dest`, `view_from`, `view_dest`:
  - same as [`exclusive_scan`](kokkosexperimentalexclusive_scan)
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1 the default string is: "Kokkos::transform_exclusive_scan_iterator_api_default"
  - for 3 the default string is: "Kokkos::transform_exclusive_scan_view_api_default"
- `unary_op`:
  - *unary* functor performing the desired transformation operation to an element.
  Must be valid to be called from the execution space passed, and callable with
  an arguments `v` of type (possible const) `value_type`,
  where `value_type` is the value type of `first_from` (for 1,2)
  or the value type of `view_from` (for 3,4), and must not modify `v`.
  - Must conform to:
  ```c++
  struct UnaryOp {
	KOKKOS_FUNCTION
	constexpr value_type operator()(const value_type & v) const {
	  return /* ... */
	}
  };

### Return

Iterator to the element *after* the last element written.

------

(kokkosexperimentalinclusive_scan)=
## `Kokkos::Experimental::inclusive_scan`

```c++
//
// overload set A
//
template <class ExecutionSpace, class InputIteratorType, class OutputIteratorType>
OutputIteratorType inclusive_scan(const ExecutionSpace& exespace,                 (1)
                                  InputIteratorType first_from,
                                  InputIteratorType last_from,
                                  OutputIteratorType first_dest);

template <class ExecutionSpace, class InputIteratorType, class OutputIteratorType>
OutputIteratorType inclusive_scan(const std::string& label,                       (2)
                                  const ExecutionSpace& exespace,
                                  InputIteratorType first_from,
                                  InputIteratorType last_from,
                                  OutputIteratorType first_dest);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2>
auto inclusive_scan(const ExecutionSpace& exespace,                               (3)
                    const ::Kokkos::View<DataType1, Properties1...>& view_from,
                    const ::Kokkos::View<DataType2, Properties2...>& view_dest);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2>
auto inclusive_scan(const std::string& label, const ExecutionSpace& exespace,     (4)
                    const ::Kokkos::View<DataType1, Properties1...>& view_from,
                    const ::Kokkos::View<DataType2, Properties2...>& view_dest);

//
// overload set B
//
template <
 class ExecutionSpace, class InputIteratorType,
 class OutputIteratorType, class BinaryOpType
 >
OutputIteratorType inclusive_scan(const ExecutionSpace& exespace,                 (5)
                                  InputIteratorType first_from,
                                  InputIteratorType last_from,
                                  OutputIteratorType first_dest,
                                  BinaryOpType bin_op);

template <
  class ExecutionSpace, class InputIteratorType,
  class OutputIteratorType, class BinaryOpType
  >
OutputIteratorType inclusive_scan(const std::string& label,                       (6)
                                  const ExecutionSpace& exespace,
                                  InputIteratorType first_from,
                                  InputIteratorType last_from,
                                  OutputIteratorType first_dest,
                                  BinaryOpType bin_op);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class BinaryOpType>
auto inclusive_scan(const ExecutionSpace& exespace,                               (7)
                    const ::Kokkos::View<DataType1, Properties1...>& view_from,
                    const ::Kokkos::View<DataType2, Properties2...>& view_dest,
                    BinaryOpType bin_op);

template <
  class ExecutionSpace, class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class BinaryOpType>
auto inclusive_scan(const std::string& label, const ExecutionSpace& exespace,     (8)
                    const ::Kokkos::View<DataType1, Properties1...>& view_from,
                    const ::Kokkos::View<DataType2, Properties2...>& view_dest,
                    BinaryOpType bin_op);

//
// overload set C
//
template <
 class ExecutionSpace,
 class InputIteratorType, class OutputIteratorType,
 class BinaryOpType, class ValueType
 >
OutputIteratorType inclusive_scan(const ExecutionSpace& exespace,                 (9)
                                  InputIteratorType first_from,
                                  InputIteratorType last_from,
                                  OutputIteratorType first_dest,
                                  BinaryOpType bin_op,
                                  ValueType init_value);

template <
  class ExecutionSpace, class InputIteratorType,
  class OutputIteratorType, class BinaryOpType, class ValueType
  >
OutputIteratorType inclusive_scan(const std::string& label,                       (10)
                                  const ExecutionSpace& exespace,
                                  InputIteratorType first_from,
                                  InputIteratorType last_from,
                                  OutputIteratorType first_dest,
                                  BinaryOpType bin_op,
                                  ValueType init_value);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class BinaryOpType, class ValueType>
auto inclusive_scan(const ExecutionSpace& exespace,                               (11)
                    const ::Kokkos::View<DataType1, Properties1...>& view_from,
                    const ::Kokkos::View<DataType2, Properties2...>& view_dest,
                    BinaryOpType bin_op,
                    ValueType init_value);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class BinaryOpType, class ValueType>
auto inclusive_scan(const std::string& label, const ExecutionSpace& exespace,     (12)
                    const ::Kokkos::View<DataType1, Properties1...>& view_from,
                    const ::Kokkos::View<DataType2, Properties2...>& view_dest,
                    BinaryOpType bin_op,
                    ValueType init_value);
```

### Description

- 1,2,3,4: computes an inclusive prefix scan over the range `[first_from, last_from)` (1,2)
or for `view_from` (3,4) using the binary op `bin_op` to combine two elements,
and writes the results to the range beginning at `first_dest` (1,2) or to `view_dest` (3,4).

- 5,6,7,8: computes an inclusive prefix scan over the range `[first_from, last_from)` (5,6)
or `view_from` (7,8) using the binary op `bin_op` to combine two elements, and writes
the results to the range beginning at `first_dest` (5,6) or to `view_dest` (7,8).

- 9,10,11,12: computes an inclusive prefix scan over the range `[first_from, last_from)` (9,10)
or `view_from` (11,12) using the binary functor `bin_op` to combine two elements
and `init` as the initial value, and writes
the results to the range beginning at `first_dest` (9,10) or to `view_dest` (11,12).

Inclusive means that the i-th input element is included in the i-th sum.

### Parameters and Requirements

- `exespace`, `first_from`, `first_last`, `first_dest`, `view_from`, `view_dest`, `bin_op`:
  - same as in [`exclusive_scan`](kokkosexperimentalexclusive_scan)
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1,5,9 the default string is: "Kokkos::inclusive_scan_iterator_api_default"
  - for 3,7,11 the default string is: "Kokkos::inclusive_scan_view_api_default"

### Return

Iterator to the element *after* the last element written.

------

(kokkosexperimentaltransform_inclusive_scan)=
## `Kokkos::Experimental::transform_inclusive_scan`

```c++
template <
  class ExecutionSpace, class InputIteratorType,
  class OutputIteratorType, class BinaryOpType, class UnaryOpType>
OutputIteratorType transform_inclusive_scan(const ExecutionSpace& exespace,       (1)
                                            InputIteratorType first_from,
                                            InputIteratorType last_from,
                                            OutputIteratorType first_dest,
                                            BinaryOpType binary_op,
                                            UnaryOpType unary_op);

template <
  class ExecutionSpace, class InputIteratorType,
  class OutputIteratorType, class BinaryOpType, class UnaryOpType>
OutputIteratorType transform_inclusive_scan(const std::string& label,             (2)
                                            const ExecutionSpace& exespace,
                                            InputIteratorType first_from,
                                            InputIteratorType last_from,
                                            OutputIteratorType first_dest,
                                            BinaryOpType binary_op,
                                            UnaryOpType unary_op);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class BinaryOpType, class UnaryOpType>
auto transform_inclusive_scan(const ExecutionSpace& exespace,                     (3)
                              const ::Kokkos::View<DataType1, Properties1...>& view_from,
                              const ::Kokkos::View<DataType2, Properties2...>& view_dest,
                              BinaryOpType binary_op,
                              UnaryOpType unary_op);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class BinaryOpType, class UnaryOpType>
auto transform_inclusive_scan(const std::string& label,                           (4)
                              const ExecutionSpace& exespace,
                              const ::Kokkos::View<DataType1, Properties1...>& view_from,
                              const ::Kokkos::View<DataType2, Properties2...>& view_dest,
                              BinaryOpType binary_op,
                              UnaryOpType unary_op);

template <
  class ExecutionSpace, class InputIteratorType,
  class OutputIteratorType, class BinaryOpType, class UnaryOpType,
  class ValueType>
OutputIteratorType transform_inclusive_scan(const ExecutionSpace& exespace,       (5)
                                            InputIteratorType first_from,
                                            InputIteratorType last_from,
                                            OutputIteratorType first_dest,
                                            BinaryOpType binary_op,
                                            UnaryOpType unary_op,
                                            ValueType init_value);

template <
  class ExecutionSpace, class InputIteratorType,
  class OutputIteratorType, class BinaryOpType, class UnaryOpType,
  class ValueType>
OutputIteratorType transform_inclusive_scan(const std::string& label,             (6)
                                            const ExecutionSpace& exespace,
                                            InputIteratorType first_from,
                                            InputIteratorType last_from,
                                            OutputIteratorType first_dest,
                                            BinaryOpType binary_op,
                                            UnaryOpType unary_op,
                                            ValueType init_value);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class BinaryOpType, class UnaryOpType, class ValueType>
auto transform_inclusive_scan(const ExecutionSpace& exespace,                     (7)
                              const ::Kokkos::View<DataType1, Properties1...>& view_from,
                              const ::Kokkos::View<DataType2, Properties2...>& view_dest,
                              BinaryOpType binary_op, UnaryOpType unary_op,
                              ValueType init_value);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class BinaryOpType, class UnaryOpType, class ValueType>
auto transform_inclusive_scan(const std::string& label,                           (8)
                              const ExecutionSpace& exespace,
                              const ::Kokkos::View<DataType1, Properties1...>& view_from,
                              const ::Kokkos::View<DataType2, Properties2...>& view_dest,
                              BinaryOpType binary_op, UnaryOpType unary_op,
                              ValueType init_value);
```

### Description

- 1,2: transforms each element in the range `[first_from, last_from)`
with `unary_op`, then computes an inclusive prefix scan operation using `binary_op`
over the resulting range, and writes the results to the range beginning at `first_dest`.

- 3,4: same as (1,2) except that the elements are read from `view_from`
and written to `view_dest`

- 5,6: same as (1,2) but the scan accounts for the `init_value`.

- 7,8: same as (3,4) but the scan accounts for the `init_value`.

Inclusive means that the i-th input element is included in the i-th sum.

### Parameters and Requirements

- `exespace`, `first_from`, `first_last`, `first_dest`, `view_from`, `view_dest`, `init_value`, `bin_op`, `unary_op`:
  - same as [`transform_exclusive_scan`](kokkosexperimentaltransform_exclusive_scan)
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1,5 the default string is: "Kokkos::transform_inclusive_scan_iterator_api_default"
  - for 3,7 the default string is: "Kokkos::transform_inclusive_scan_view_api_default"

### Return

Iterator to the element *after* the last element written.

------

(kokkosexperimentaladjant_difference)=
## `Kokkos::Experimental::adjacent_difference`

```c++
template <class ExecutionSpace, class InputIteratorType, class OutputIteratorType>
OutputIteratorType adjacent_difference(const ExecutionSpace& exespace,                    (1)
                                       InputIteratorType first_from,
                                       InputIteratorType last_from,
                                       OutputIteratorType first_dest);

template <class ExecutionSpace, class InputIteratorType, class OutputIteratorType, class BinaryOp>
OutputIteratorType adjacent_difference(const ExecutionSpace& exespace,                    (2)
                                       InputIteratorType first_from,
                                       InputIteratorType last_from,
                                       OutputIteratorType first_dest,
                                       BinaryOp bin_op);

template <class ExecutionSpace, class InputIteratorType, class OutputIteratorType>
OutputIteratorType adjacent_difference(const std::string& label,                          (3)
                                       const ExecutionSpace& exespace,
                                       InputIteratorType first_from,
                                       InputIteratorType last_from,
                                       OutputIteratorType first_dest);

template <class ExecutionSpace, class InputIteratorType, class OutputIteratorType, class BinaryOp>
OutputIteratorType adjacent_difference(const std::string& label,                          (4)
                                       const ExecutionSpace& exespace,
                                       InputIteratorType first_from,
                                       InputIteratorType last_from,
                                       OutputIteratorType first_dest,
                                       BinaryOp bin_op);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2>
auto adjacent_difference(const ExecutionSpace& exespace,                                  (5)
                         const ::Kokkos::View<DataType1, Properties1...>& view_from,
                         const ::Kokkos::View<DataType2, Properties2...>& view_dest);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class BinaryOp>
auto adjacent_difference(const ExecutionSpace& exespace,                                  (6)
                         const ::Kokkos::View<DataType1, Properties1...>& view_from,
                         const ::Kokkos::View<DataType2, Properties2...>& view_dest,
                         BinaryOp bin_op);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2>
auto adjacent_difference(const std::string& label,                                        (7)
                         const ExecutionSpace& exespace,
                         const ::Kokkos::View<DataType1, Properties1...>& view_from,
                         const ::Kokkos::View<DataType2, Properties2...>& view_dest);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class DataType2, class... Properties2,
  class BinaryOp>
auto adjacent_difference(const std::string& label,                                        (8)
                         const ExecutionSpace& exespace,
                         const ::Kokkos::View<DataType1, Properties1...>& view_from,
                         const ::Kokkos::View<DataType2, Properties2...>& view_dest,
                         BinaryOp bin_op);
```

### Description

- (1,3,5,7): First, a copy of `*first_from` is written to `*first_dest` for (1,3),
  or a copy of `view_from(0)` is written to `view_dest(0)` for (5,7).
  Second, it computes the *difference* between the second and the first
  of each adjacent pair of elements of the range `[first_from, last_from)` for (1,3)
  or in `view_from` for (5,7), and writes them to the range beginning at `first_dest + 1` for (1,3),
  or `view_dest` for (5,7).

- (2,4,6,8): First, a copy of `*first_from` is written to `*first_dest` for (2,4),
  or a copy of `view_from(0)` is written to `view_dest(0)` for (6,8).
  Second, it calls the binary functor with the second and the first elements
  of each adjacent pair of elements of the range `[first_from, last_from)` for (2,4)
  or in `view_from` for (6,8), and writes them to the range beginning at `first_dest + 1` for (2,4),
  or `view_dest` for (6,8).


### Parameters and Requirements

- `exespace`:
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1,2 the default string is: "Kokkos::adjacent_difference_iterator_api_default"
  - for 5,6 the default string is: "Kokkos::adjacent_difference_view_api_default"
- `first_from`, `last_from`, `first_dest`:
  - range of elements to read from `*_from` and write to `first_dest`
  - must be *random access iterators*
  - must represent a valid range, i.e., `last_from >= first_from` (checked in debug mode)
  - must be accessible from `exespace`
- `view_from`, `view_dest`:
  - views to read elements from `view_from` and write to `view_dest`
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`
- `bin_op`:
  - *binary* functor representing the operation to apply to each pair of elements.
  Must be valid to be called from the execution space passed, and callable with
  two arguments `a,b` of type (possible const) `value_type`, where `value_type`
  is the value type of `InputIteratorType` (for 1,2,3,4) or the value type
  of `view_from` (for 5,6,7,8), and must not modify `a,b`.
  - must conform to:
  ```c++
  struct BinaryOp
  {
     KOKKOS_INLINE_FUNCTION
     return_type operator()(const value_type & a,
	                        const value_type & b) const {
       return /* ... */;
     }

     // or, also valid
     return_type operator()(value_type a,
	                        value_type b) const {
       return /* ... */;
     }
  };
  ```
  The return type `return_type` must be such that an object of type `OutputIteratorType` for (1,2,3,4)
  or an object of type `value_type` where `value_type` is the value type of `view_dest` for (5,6,7,8)
  can be dereferenced and assigned a value of type `return_type`.

### Return

Iterator to the element *after* the last element written.
