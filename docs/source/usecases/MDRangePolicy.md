# MDRangePolicy Use Case

## Operations on multi-dimensional arrays

This example demonstrates usage of the [`MDRangePolicy`](../API/core/policies/MDRangePolicy) in the context of performing operations on multidimensional arrays or tensor data.
Such a use case occurs for example when working on numerical solution to PDEs such as finite element methods, where discretization of the space may result in `C` cells (elements), and definition of basis functions that take `P` evaluation points as input arguments and return output whose rank and dimensions `D` depend on the field rank `F` of the basis function.


## Problem formulation

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

The most straightforward way to parallelize the serial code above is to convert the outer `for` loop over cells with the sequential iteration pattern into a parallel for loop using a [`RangePolicy`](../API/core/policies/RangePolicy)

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

There is more parallelism to exploit, particularly within the for loops over fields `F` and points `P`. One way to accomplish this would involve taking the product of the three iteration ranges, `C*F*P`, and performing a [`parallel_for`](../API/core/parallel-dispatch/parallel_for) over that product. However, this would require extraction routines to map between indices from the flattened iteration range, `C*F*P`, and the multidimensional indices required by data structures in this example. In addition, to achieve performance portability the mapping between the 1-D product iteration range and multidimensional 3-D indices would require architecture-awareness, akin to the notion of [`LayoutLeft`](../API/core/view/layoutLeft) and [`LayoutRight`](../API/core/view/layoutRight) used in Kokkos to establish data access patterns.

The [`MDRangePolicy`](../API/core/policies/MDRangePolicy) provides a natural way to accomplish the goal of parallelize over all three iteration ranges without requiring manually computing the product of the iteration ranges and mapping between 1-D and 3-D multidimensional indices. The [`MDRangePolicy`](../API/core/policies/MDRangePolicy) is suitable for use with tightly-nested for loops and provides a method to expose additional parallelism in computations beyond simply parallelize in a single dimension, as was shown in the first implementation using the [`RangePolicy`](../API/core/policies/RangePolicy).

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

The [`MDRangePolicy`](../API/core/policies/MDRangePolicy) accepts the same template parameters as the [`RangePolicy`](../API/core/policies/RangePolicy), but also requires an additional type - the `Kokkos::Rank<R>` parameter, where `R` is the rank, that is the number of nested for-loops, and must be provided at compile-time.

The policy requires two arguments:
  1) An initializer list, or `Kokkos::Array`, of "begin" indices
  2) An initializer list, or `Kokkos::Array`, of "end" indices

Internally the [`MDRangePolicy`](../API/core/policies/MDRangePolicy) uses tiling over the multidimensional iteration space. For customization an optional third argument may be passed to the policy - an initializer list of tile dimension sizes. This argument might become important when performance tuning, as simple default sizes can be problem-dependent and difficult to determine automatically.

The signature of the lambda (or access operator of the functor) requires an argument for each rank.

The [`MDRangePolicy`](../API/core/policies/MDRangePolicy) can be used with both the [`parallel_for`](../API/core/parallel-dispatch/parallel_for) and [`parallel_reduce`](../API/core/parallel-dispatch/parallel_reduce) patterns in Kokkos.


## References

The API reference for the [`MDRangePolicy`](../API/core/policies/MDRangePolicy) is available on the Kokkos wiki:
  [wiki link](https://github.com/kokkos/kokkos/wiki/Kokkos%3A%3AMDRangePolicy)
 
The use case that this example is based on comes from the Intrepid2 package of Trilinos. For more examples, check out code in Trilinos in files at: `Trilinos/packages/intrepid2/src/Shared/Intrepid2_ArrayToolsDef*.hpp`.

This link provides some overview of the Intrepid package: 
  [documentation link](https://trilinos.org/packages/intrepid/)

