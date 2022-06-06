# `MaxLoc`

Specific implementation of [ReducerConcept](Kokkos%3A%3AReducerConcept) storing the maximum value

Header File: `Kokkos_Core.hpp`

Usage: 
  ```c++
  MaxLoc<T,I,S>::value_type result;
  parallel_reduce(N,Functor,MaxLoc<T,I,S>(result));
  ```

. 

## Synopsis 
  ```c++
  template<class Scalar, class Index, class Space>
  class MaxLoc{
    public:
      typedef MaxLoc reducer;
      typedef ValLocScalar<typename std::remove_cv<Scalar>::type,
                           typename std::remove_cv<Index>::type > value_type;
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
      MaxLoc(value_type& value_);

      KOKKOS_INLINE_FUNCTION
      MaxLoc(const result_view_type& value_);
  };
  ```

## Public Class Members

### Typedefs
   
 * `reducer`: The self type.
 * `value_type`: The reduction scalar type (specialization of [ValLocScalar](Kokkos%3A%3AValLocScalar))
 * `result_view_type`: A `Kokkos::View` referencing the reduction result 

### Constructors
 
 * ```c++
   MaxLoc(value_type& result)
   ```
   Constructs a reducer which references a local variable as its result location.  
 
 * ```c++
   MaxLoc(const result_view_type result)`
   ```
   Constructs a reducer which references a specific view as its result location.

### Functions

 * ```c++
   void join(value_type& dest, const value_type& src)  const;`
   ```
   Store maximum with index of `src` and `dest` into `dest`:  `dest = (src.val > dest.val) ? src : dest;`. 

 * ```c++
   void join(volatile value_type& dest, const volatile value_type& src) const;
   ```
   Store maximum with index of `src` and `dest` into `dest`:  `dest = (src.val > dest.val) ? src : dest;`. 

 * ```c++
   void init( value_type& val)  const;
   ```
   Initialize `val.val` using the Kokkos::reduction_identity<Scalar>::max() method.  The default implementation sets `val=<TYPE>_MIN`.

   Initialize `val.loc` using the Kokkos::reduction_identity<Index>::min() method.  The default implementation sets `val=<TYPE>_MAX`.

 * ```c++
   value_type& reference() const;
   ```
   Returns a reference to the result provided in class constructor.

 * ```c++
   result_view_type view() const;
   ```
   Returns a view of the result place provided in class constructor.

### Additional Information
   * `MaxLoc<T,I,S>::value_type` is Specialization of ValLocScalar on non-const `T` and non-const `I`
   * `MaxLoc<T,I,S>::result_view_type` is `Kokkos::View<T,S,Kokkos::MemoryTraits<Kokkos::Unmanaged>>`.  Note that the S (memory space) must be the same as the space where the result resides.
   * Requires: `Scalar` has `operator =` and `operator >` defined. `Kokkos::reduction_identity<Scalar>::max()` is a valid expression. 
   * Requires: `Index` has `operator =` defined. `Kokkos::reduction_identity<Index>::min()` is a valid expression. 
   * In order to use MaxLoc with a custom type of either `Scalar` or `Index`, a template specialization of Kokkos::reduction_identity<CustomType> must be defined.  See [Using Built-in Reducers with Custom Scalar Types](Custom-Reductions%3A-Built-In-Reducers-with-Custom-Scalar-Types) for details