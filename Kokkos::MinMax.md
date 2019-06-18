# `Kokkos::MinLoc`

Specific implementation of [ReducerConcept](Kokkos%3A%3AReducerConcept) storing both the minimum and maximum values

Header File: `Kokkos_Core.hpp`

Usage: 
  ```c++
  MinMax<T,S>::value_type result;
  parallel_reduce(N,Functor,MinLoc<T,S>(result));
  ```

. 

## Synopsis 
  ```c++
  template<class Scalar, class Space>
  class MinMax{
    public:
      typedef MinMax reducer;
      typedef MinMaxScalar<typename std::remove_cv<Scalar>::type> value_type;
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
      MinMax(value_type& value_);

      KOKKOS_INLINE_FUNCTION
      MinMax(const result_view_type& value_);
  };
  ```

## Public Class Members

### Typedefs
   
 * `reducer`: The self type.
 * `value_type`: The reduction scalar type (specialization of [MinMaxScalar](Kokkos%3A%3AMinMaxScalar))
 * `result_view_type`: A `Kokkos::View` referencing the reduction result 

### Constructors
 
 * ```c++
   MinMax(value_type& result)
   ```
   Constructs a reducer which references a local variable as its result location.  
 
 * ```c++
   MinMax(const result_view_type result)
   ```
   Constructs a reducer which references a specific view as its result location.

### Functions

 * ```c++
   void join(value_type& dest, const value_type& src)  const;
   ```
   - Store minimum of `src` and `dest` into `dest`:  `dest.min_val = (src.min_val < dest.min_val) ? src.min_val : dest.min_val;`.
   - Store maximum of `src` and `dest` into `dest`:  `dest.max_val = (src.max_val < dest.max_val) ? src.max_val : dest.max_val;`.
 * ```c++
   void join(volatile value_type& dest, const volatile value_type& src) const;
   ```
    ** Store minimum of `src` and `dest` into `dest`:  `dest.min_val = (src.min_val < dest.min_val) ? src.min_val : dest.min_val;`.
   ** Store maximum of `src` and `dest` into `dest`:  `dest.max_val = (src.max_val < dest.max_val) ? src.max_val : dest.max_val;`. 

 * ```c++
   void init( value_type& val)  const;
   ```
   Initialize `val.min_val` using the Kokkos::reduction_identity<Scalar>::min() method.  The default implementation sets `val=<TYPE>_MAX`.

   Initialize `val.max_val` using the Kokkos::reduction_identity<Index>::max() method.  The default implementation sets `val=<TYPE>_MIN`.

 * ```c++
   value_type& reference() const;
   ```
   Returns a reference to the result provided in class constructor.

 * ```c++
   result_view_type view() const;
   ```
   Returns a view of the result place provided in class constructor.

### Additional Information
   * `MinMax<T,S>::value_type` is Specialization of MinMaxScalar on non-const `T`
   * `MinLoc<T,S>::result_view_type` is `Kokkos::View<T,S,Kokkos::MemoryTraits<Kokkos::Unmanaged>>`.  Note that the S (memory space) must be the same as the space where the result resides.
   * Requires: `Scalar` has `operator =`, `operator <` and `operator >` defined. `Kokkos::reduction_identity<Scalar>::min()` and `Kokkos::reduction_identity<Scalar>::max()` are a valid expressions. 
   * In order to use MinMax with a custom type of `Scalar`, a template specialization of Kokkos::reduction_identity<CustomType> must be defined.  See [Using Built-in Reducers with Custom Scalar Types](Custom-Reductions%3A-Built-In-Reducers-with-Custom-Scalar-Types) for details