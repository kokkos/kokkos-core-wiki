# `Kokkos::Sum`

Specific implementation of ReducerConcept(Kokkos%3A%3AReducerConcept) performing an `add` operation

Header File: `Kokkos_Core.hpp`

Usage: 
  ```c++
  T result;
  parallel_reduce(N,Functor,Sum<T,S>(result));
  ```

. 

## Synopsis 
  ```c++
  template<class Scalar, class Space>
  class Sum{
    public:
      //Required for Concept
      typedef Sum reducer;
      typedef typename std::remove_cv<Scalar>::type value_type;
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

      
      //Part of Build-In reducers for Kokkos
      KOKKOS_INLINE_FUNCTION
      Reducer(value_type& value_);

      KOKKOS_INLINE_FUNCTION
      Reducer(const result_view_type& value_);
  };
  ```

## Public Class Members

### Typedefs
   
 * `reducer`: The self type.
 * `value_type`: The reduction scalar type.
 * `result_view_type`: A `Kokkos::View` referencing where the reduction result 

### Constructors
 
 * ```c++
   Reducer(value_type& result)
   ```
   Constructs a reducer which references a local variable as its result location.  
 
 * ```c++
   Reducer(const result_view_type result)`
   ```
   Constructs a reducer which references a specific view as its result location.

### Functions

 * ```c++
   void join(value_type& dest, const value_type& src)  const;`
   ```
   Add `src` into `dest`:  `dest+=src;`. 

 * ```c++
   void join(volatile value_type& dest, const volatile value_type& src) const;
   ```
   Add `src` into `dest`: `dest+=src;`. 

 * ```c++
   void init( value_type& val)  const;
   ```
   Initialize `val` with `0` value.    

 * ```c++
   value_type& reference() const;
   ```
   Returns a reference to the result place.

 * ```c++
   result_view_type view() const;
   ```
   Returns a view of the result place. 

### Additional Information
   * `Sum<T,S>::value_type` is `T`
   * `Sum<T,S>::result_view_type` is `Kokkos::View<T,S,Kokkos::MemoryTraits<Kokkos::Unmanaged>>`
   * Requires: `Scalar` has `operator =` and `operator +=` defined. `Kokkos::reduction_identity<Scalar>::sum()` is a valid expression. 
   * In order to use Sum with a custom type, a template specialization of Kokkos::reduction_identity<CustomType> must be defined.  See 