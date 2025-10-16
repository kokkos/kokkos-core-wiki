# Fortran Interop Use Case

## Operations on multidimensional fortran allocated arrays using Kokkos 

This example demonstrates usage of the `Fortran Language Compatibility Layer (FLCL)` in the context of performing a DAXPY (double precision A * X + Y) using Kokkos from a simple Fortran program. Such a use case occurs when using Kokkos for performance portability within a Fortran application. 

## Program structure 
This example uses the Kokkos Fortran interop utilities in [FLCL](https://github.com/kokkos/kokkos-fortran-interop). 
This includes a set of Fortran routines for converting Fortran allocated arrays into a ndarray and a set of C++ functions for converting a ndarray into a Kokkos unmanaged view. 

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

To convert a Fortran allocated array into a ndarray we use a set of procedures (behind an interface) defined in [flcl-f.f90](https://github.com/kokkos/kokkos-fortran-interop/blob/master/src/flcl-f.f90)

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

To convert a ndarray to a Kokkos::View we use view_from_ndarray defined in [flcl-cxx.hpp](https://github.com/kokkos/kokkos-fortran-interop/blob/master/src/flcl-cxx.hpp)
``` c++ 
template <typename DataType>
  Kokkos::View<DataType, Kokkos::LayoutStride, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
  view_from_ndarray(flcl_ndarray_t const &ndarray) 
```

These are the main utilities that will be used in our DAXPY example. 

We begin with a Fortran program defined in [axpy-ndarray-main.f90](https://github.com/kokkos/kokkos-fortran-interop/blob/master/examples/01-axpy-ndarray/axpy-ndarray-main.F90)

We start by bringing in the flcl module: 
``` fortran
use :: flcl_mod
```
We then define our arrays including two 'Y' arrays, one will be used for calculating the daxpy result with Fortran, the other with kokkos.
``` fortran 
  real(c_double), dimension(:), allocatable :: f_y
  real(c_double), dimension(:), allocatable :: c_y
  real(c_double), dimension(:), allocatable :: x
  real(c_double) :: alpha
``` 
Performing the DAXPY in Fortran is simply: 
``` fortran 
do ii = 1, mm
    f_y(ii) = f_y(ii) + alpha * x(ii)
end do
``` 

Performing the DAXPY in Kokkos begins with a call to axpy: 
``` fortran 
call axpy(c_y, x, alpha)
``` 

This is defined in [axpy-ndarray-f.f90](https://github.com/kokkos/kokkos-fortran-interop/blob/master/examples/01-axpy-ndarray/axpy-ndarray-f.f90)
``` fortran 
subroutine axpy( y, x, alpha )
   use, intrinsic :: iso_c_binding
   use :: flcl_mod
   implicit none
   real(c_double), dimension(:), intent(inout) :: y
   real(c_double), dimension(:), intent(in) :: x
   real(c_double), intent(in) :: alpha

   call f_axpy(to_nd_array(y), to_nd_array(x), alpha)
end subroutine axpy
```
Which calls the subroutine f_axpy but prior to doing so converts the Fortran arrays into nd_arrays. 
f_axpy is defined earlier and note that f_axpy is bound to the C routine 'c_axpy'. 
``` fortran
interface
    subroutine f_axpy( nd_array_y, nd_array_x, alpha ) &
        & bind(c, name='c_axpy')
        use, intrinsic :: iso_c_binding
        use :: flcl_mod
        type(nd_array_t) :: nd_array_y
        type(nd_array_t) :: nd_array_x
        real(c_double) :: alpha
    end subroutine f_axpy
end interface
```

c_axpy is where we make use of Kokkos for the computation and is defined in [axpy-ndarray-cxx.cc](https://github.com/kokkos/kokkos-fortran-interop/blob/master/examples/01-axpy-ndarray/axpy-ndarray-cxx.cc)

```c++ 
void c_axpy( flcl_ndarray_t *nd_array_y, flcl_ndarray_t *nd_array_x, double *alpha ) {
  using flcl::view_from_ndarray;

  auto y = view_from_ndarray<double*>(*nd_array_y);
  auto x = view_from_ndarray<double*>(*nd_array_x);

  Kokkos::parallel_for( "axpy", y.extent(0), KOKKOS_LAMBDA( const size_t idx)
  {
    y(idx) += *alpha * x(idx);
  });

  return;
}
```

In this function we first convert our two nd_array to [`Kokkos::View`](../API/core/view/view) and then use [`Kokkos::parallel_for`](../API/core/parallel-dispatch/parallel_for) with a simply DAXPY lambda.  

This use case illustrates the ability to use Kokkos in Fortran applications with interoperability of Fortran arrays and [`Kokkos::View`](../API/core/view/view) via the ndarray type and conversion routines provided in FLCL. 
