# `MinMaxLoc`

Specific implementation of [ReducerConcept](ReducerConcept) storing both the minimum and maximum values with corresponding indicies

Header File: `Kokkos_Core.hpp`

Usage: 
```c++
MinMaxLoc<T,I,S>::value_type result;
parallel_reduce(N,Functor,MinMaxLoc<T,I,S>(result));
```

## Synopsis 
```c++
template<class Scalar, class Space>
class MinMaxLoc{
 public:
   typedef MinMaxLoc reducer;
   typedef MinMaxLocScalar<typename std::remove_cv<Scalar>::type,
                           typename std::remove_cv<Index>::type> value_type;
   typedef Kokkos::View<value_type, Space> result_view_type;
   
   KOKKOS_INLINE_FUNCTION
   void join(value_type& dest, const value_type& src)  const

   KOKKOS_INLINE_FUNCTION
   void join(volatile value_type& dest, const volatile value_type& src) const;

   KOKKOS_INLINE_FUNCTION
   void init( value_type& val)  const;

   KOKKOS_INLINE_FUNCTION
   value_type& reference() const;

   KOKKOS_INLINE_FUNCTION
   result_view_type view() const;

   KOKKOS_INLINE_FUNCTION
   MinMaxLoc(value_type& value_);

   KOKKOS_INLINE_FUNCTION
   MinMaxLoc(const result_view_type& value_);
};
```

## Public Class Members

### Typedefs
   
 * `reducer`: The self type.
 * `value_type`: The reduction scalar type (specialization of [MinMaxLocScalar](MinMaxLocScalar))
 * `result_view_type`: A `Kokkos::View` referencing the reduction result 

### Constructors
 
 * ```c++
   MinMaxLoc(value_type& result)
   ```
   Constructs a reducer which references a local variable as its result location.  
 
 * ```c++
   MinMaxLoc(const result_view_type result)
   ```
   Constructs a reducer which references a specific view as its result location.

### Functions

 * ```c++
   void join(value_type& dest, const value_type& src)  const;
   ```
   - Store minimum with location of `src` and `dest` into `dest`.
   - Store maximum with location of `src` and `dest` into `dest`.
 * ```c++
   void join(volatile value_type& dest, const volatile value_type& src) const;
   ```
    - Store minimum with location of `src` and `dest` into `dest`.
   - Store maximum with location of `src` and `dest` into `dest`. 

 * ```c++
   void init( value_type& val)  const;
   ```
   Initialize `val.min_val` using the Kokkos::reduction_identity<Scalar>::min() method.  The default implementation sets `val=<TYPE>_MAX`.

   Initialize `val.max_val` using the Kokkos::reduction_identity<Index>::max() method.  The default implementation sets `val=<TYPE>_MIN`.

   Initialize `val.min_loc` using the Kokkos::reduction_identity<Scalar>::min() method.  The default implementation sets `val=<TYPE>_MAX`.

   Initialize `val.max_loc` using the Kokkos::reduction_identity<Index>::min() method.  The default implementation sets `val=<TYPE>_MAX`.

 * ```c++
   value_type& reference() const;
   ```
   Returns a reference to the result provided in class constructor.

 * ```c++
   result_view_type view() const;
   ```
   Returns a view of the result place provided in class constructor.

### Additional Information
   * `MinMaxLoc<T,I,S>::value_type` is Specialization of MinMaxLocScalar on non-const `T` and non-const `I`
   * `MinMaxLoc<T,I,S>::result_view_type` is `Kokkos::View<T,S,Kokkos::MemoryTraits<Kokkos::Unmanaged>>`.  Note that the S (memory space) must be the same as the space where the result resides.
   * Requires: `Scalar` has `operator =`, `operator <` and `operator >` defined. `Kokkos::reduction_identity<Scalar>::min()` and `Kokkos::reduction_identity<Scalar>::max()` are a valid expressions. 
   * Requires: `Index` has `operator =` defined. `Kokkos::reduction_identity<Scalar>::min()` is a valid expressions.
   * In order to use MinMaxLoc with a custom type of either `Scalar` or `Index`, a template specialization of `Kokkos::reduction_identity<CustomType>` must be defined.  See [Built-In Reducers with Custom Scalar Types](../../../ProgrammingGuide/Custom-Reductions-Built-In-Reducers-with-Custom-Scalar-Types) for details
