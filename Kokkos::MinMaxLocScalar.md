# `Kokkos::MinMaxLocScalar`

Template class for storing the min and max values with indices for min/max location reducers.  Should be accessed via ::value_type defined for particular reducer.

Header File: `Kokkos_Core.hpp`

Usage: 
  ```c++
  MinMaxLoc<T,I,S>::value_type result;
  parallel_reduce(N,Functor,MinMaxLoc<T,I,S>(result));
  T minValue = result.min_val;
  T maxValue = result.max_val;
  I minLoc = result.min_loc;
  I maxLoc = result.max_loc;
  ```
. 

## Synopsis 
  ```c++
  template<class Scalar>
  struct MinMaxLocScalar{
      Scalar min_val;
      Scalar max_val;
      Index min_loc;
      Index max_loc;

      void operator = (const MinMaxLocScalar& rhs);
      void operator = (const volatile MinMaxLocScalar& rhs);
  };

## Public Members

### Variables
   
 * `min_val`: Scalar minimum Value.
 * `max_val`: Scalar maximum Value.
 * `min_loc`: minimum location(Index).
 * `max_loc`: maximum location(Index).

### Assignment operators

 * `void operator = (const MinMaxLocScalar& rhs);` 
      assign `min_val`, `max_val`, `min_loc` and `max_loc` from `rhs`;

 * `void operator = (const volatile MinMaxLocScalar& rhs);` 
      assign `min_val`, `max_val`, `min_loc` and `max_loc` from `rhs`;