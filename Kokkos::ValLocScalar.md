# `Kokkos::ValLocScalar`

Template class for storing a value plus index for min/max location reducers.  Should be accessed via ::value_type defined for particular reducer.

Header File: `Kokkos_Core.hpp`

Usage: 
  ```c++
  MaxLoc<T,I,S>::value_type result
  parallel_reduce(N,Functor,MaxLoc<T,I,S>(result));
  T resultValue = result.val;
  I resultIndex = result.loc;
  ```
. 

## Synopsis 
  ```c++
  template<class Scalar, class Index>
  struct ValLocScalar{
      Scalar val;
      Index loc;

      void operator = (const ValLocScalar& rhs);
      void operator = (const volatile ValLocScalar& rhs);
  };

## Public Members

### Variables
   
 * `val`: Scalar Value.
 * `loc`: Scalar index.

### Assignment operators

 * `void operator = (const ValLocScalar& rhs);` 
      assign `val` and `loc` from `rhs`;

 * `void operator = (const volatile ValLocScalar& rhs);` 
      assign `val` and `loc` from `rhs`;