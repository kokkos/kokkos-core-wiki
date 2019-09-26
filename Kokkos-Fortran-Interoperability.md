# Fortran Interop Use Case

## Operations on multi-dimensional fortran allocated arrays using Kokkos 

This example demonstrates usage of the `teh Fortran Language Compatibility Layer (FLCL)` in the context of performing a simple DAXPY (double precision A * X + Y) using Kokkos from a simple fortran program. Such a use case occurs when using Kokkos for for performance portability within a Fortran application. 

## Program structure 
This example uses the Kokkos fortran interop utilities in [FLCL](https://github.com/kokkos/kokkos-fortran-interop). 
This includes a set of fortran routines for converting fortran allocated arrays into an ndarray and a set of C++ functions for converting an ndarray into a Kokkos unmanaged view. 

The ndarray type (flcl_ndarray_t) is a simple struct that captures the rank, dimensions, strides (equivalent to a dope vector) along with the flattened data. This is defined and implemented in [flcl-cxx.hpp](https://github.com/kokkos/kokkos-fortran-interop/blob/master/src/flcl-cxx.hpp)

```c++ 

typedef struct _flcl_nd_array_t {
    size_t rank;
    size_t dims[FLCL_NDARRAY_MAX_RANK];
    size_t strides[FLCL_NDARRAY_MAX_RANK];
    void *data;
} flcl_ndarray_t;

```
This has a fortran equivalent type located in [flcl-f.f90](https://github.com/kokkos/kokkos-fortran-interop/blob/master/src/flcl-f.f90)

``` fortran
type, bind(C) :: nd_array_t
    integer(c_size_t) :: rank
    integer(c_size_t) :: dims(ND_ARRAY_MAX_RANK)
    integer(c_size_t) :: strides(ND_ARRAY_MAX_RANK)
    type(c_ptr) :: data
  end type nd_array_t
```

To convert a fortran allocated array into an ndarray we use a set of procedures (behind an interface) defined in [flcl-f.f90](https://github.com/kokkos/kokkos-fortran-interop/blob/master/src/flcl-f.f90)

```fortran

interface to_nd_array
    ! 1D specializations
    module procedure to_nd_array_l_1d
    module procedure to_nd_array_i32_1d
    module procedure to_nd_array_i64_1d
    module procedure to_nd_array_r32_1d
    module procedure to_nd_array_r64_1d
    
    ! 2D specializations
    module procedure to_nd_array_l_2d
    module procedure to_nd_array_i32_2d
    module procedure to_nd_array_i64_2d
    module procedure to_nd_array_r32_2d
    module procedure to_nd_array_r64_2d

    ! 3D specializations
    module procedure to_nd_array_l_3d
    module procedure to_nd_array_i32_3d
    module procedure to_nd_array_i64_3d
    module procedure to_nd_array_r32_3d
    module procedure to_nd_array_r64_3d
```

To convert an ndarray to a Kokkos::View we us view_from_ndarray defined in [flcl-cxx.hpp](https://github.com/kokkos/kokkos-fortran-interop/blob/master/src/flcl-cxx.hpp)
``` c++ 
template <typename DataType>
  Kokkos::View<DataType, Kokkos::LayoutStride, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
  view_from_ndarray(flcl_ndarray_t const &ndarray) 
```

These are the main utilities that will be used in our DAXPY example. 

We begin with a fortran program defined in [axpy-main.f90](https://github.com/kokkos/kokkos-fortran-interop/blob/master/examples/01-axpy/axpy-main.f90)

We start by bringing in the flcl module: 
``` fortran
use :: flcl_mod
```
We then define our arrays including two 'Y' arrays one will be used for calculating the daxpy result with fortran, the other with kokkos. 
``` fortran 
  real(c_double), dimension(:), allocatable :: f_y
  real(c_double), dimension(:), allocatable :: c_y
  real(c_double), dimension(:), allocatable :: x
  real(c_double) :: alpha
``` 


**Input**:
  `inputData(C,P,D,D)` - a rank 4 View
  `inputField(C,F,P,D)` - a rank 4 View


**Return**:
  `outputField(C,F,P,D)` - a rank 4 View


**Computation**: 
  For each triple in `C,F,P` compute an output field from the two input views:
  
``` c++
  for each (c,f,p) in (C,F,P)
    compute the product inputData(c,p,:,:) * inputField(c,f,p,:)
    store result in outputField(c,f,p,:)
```


## Serial implementation

``` c++

for (int c = 0; c < C; ++c)
for (int f = 0; f < F; ++f)
for (int p = 0; p < P; ++p)
{

  auto result = Kokkos::subview(outputField, c, f, p, Kokkos::ALL);
  auto left   = Kokkos::subview(inputData, c, p, Kokkos::ALL, Kokkos::ALL);
  auto right  = Kokkos::subview(inputField, c, f, p, Kokkos::ALL);
  
  for (int i=0;i<D;++i) {
  
    double tmp(0);
    
    for (int j=0;j<D;++j)
      tmp += left(i, j)*right(j);
    
    result(i) = tmp;
  }
}

```


## Parallelization with Kokkos

### Initial implementation - `RangePolicy`

The most straightforward way to parallize the serial code above is to convert the outer `for` loop over cells with the sequential iteration pattern into a parallel for loop using a `RangePolicy`


``` c++

Kokkos::parallel_for("for_all_cells", 
  Kokkos::RangePolicy<>(0,C),
   KOKKOS_LAMBDA (const int c) {
     for (int f = 0; f < F; ++f)
     for (int p = 0; p < P; ++p)
     {

      auto result = Kokkos::subview(outputField, c, f, p, Kokkos::ALL);
      auto left   = Kokkos::subview(inputData, c, p, Kokkos::ALL, Kokkos::ALL);
      auto right  = Kokkos::subview(inputField, c, f, p, Kokkos::ALL);
  
      for (int i=0;i<D;++i) {
  
        double tmp(0);
    
        for (int j=0;j<D;++j)
          tmp += left(i, j)*right(j);
    
        result(i) = tmp;
      }
     }
  });

```


If the number of cells is large enough to merit parallelization, that is the overhead for parallel dispatch plus computation time is less than total serial execution time, then the simple implementation above will result in improved performance.

There is more parallelism to exploit, particularly within the for loops over fields `F` and points `P`. One way to accomplish this would involve taking the product of the three iteration ranges, `C*F*P`, and performing a `parallel_for` over that product. However, this would require extraction routines to map between indices from the flattened iteration range, `C*F*P`, and the multidimensional indices required by data structures in this example. In addition, to achieve performance portability the mapping between the 1-D product iteration range and multi-dimensional 3-D indices would require architecture-awareness, akin to the notion of `LayoutLeft` and `LayoutRight` used in Kokkos to establish data access patterns.

The `MDRangePolicy` provides a natural way to accomplish the goal of parallelizing over all three iteration ranges without requiring manually computing the product of the iteration ranges and mapping between 1-D and 3-D multidimensional indices. The `MDRangePolicy` is suitable for use with tightly-nested for loops and provides a method to expose additional parallelism in computations beyond simply parallelizing in a single dimension, as was shown in the first implementation using the `RangePolicy`.


### Implementation - `MDRangePolicy`

``` c++

Kokkos::parallel_for("mdr_for_all_cells", 
  Kokkos::MDRangePolicy< Kokkos::Rank<3> > ({0,0,0}, {C,F,P}),
   KOKKOS_LAMBDA (const int c, const int f, const int p) {
    auto result = Kokkos::subview(outputField, c, f, p, Kokkos::ALL);
    auto left   = Kokkos::subview(inputData, c, p, Kokkos::ALL, Kokkos::ALL);
    auto right  = Kokkos::subview(inputField, c, f, p, Kokkos::ALL);
  
    for (int i=0;i<D;++i) {
  
      double tmp(0);
    
      for (int j=0;j<D;++j)
        tmp += left(i, j)*right(j);
    
      result(i) = tmp;
    }
  });

```


## MDRangePolicy usage

The `MDRangePolicy` accepts the same template parameters as the `RangePolicy`, but also requires an additional type - the `Kokkos::Rank<R>` parameter, where `R` is the rank, that is the number of nested for-loops, and must be provided at compile-time.

The policy requires two arguments:
  1) An initializer list, or `Kokkos::Array`, of "begin" indices
  2) An initializer list, or `Kokkos::Array`, of "end" indices

Internally the `MDRangePolicy` uses tiling over the multi-dimensional iteration space. For customization an optional third argument may be passed to the policy - an initializer list of tile dimension sizes. This argument might become important when performance tuning, as simple default sizes can be problem-dependent and difficult to determine automatically.

The signature of the lambda (or access operator of the functor) requires an argument for each rank.

The `MDRangePolicy` can be used with both the `parallel_for` and `parallel_reduce` patterns in Kokkos.


## References

The API reference for the `MDRangePolicy` is available on the Kokkos wiki:
  [wiki link](https://github.com/kokkos/kokkos/wiki/Kokkos%3A%3AMDRangePolicy)
 
The use case that this example is based on comes from the Intrepid2 package of Trilinos. For more examples, check out code in Trilinos in files at: `Trilinos/packages/intrepid2/src/Shared/Intrepid2_ArrayToolsDef*.hpp`.

This link provides some overview of the Intrepid package: 
  [documentation link](https://trilinos.org/packages/intrepid/)

