In order to use a Custom Scalar Type with Built-in reductions, the following requirements must be fulfilled.

   * An initialization function must be provided via a specialization of the Kokkos::reduction_identity<T> class.  
   * Operators required for applied reduction class must be implemented.
   * The class / struct must either use the default copy constructor or have a specific copy constructor 
     implemented. 

## Example

```c++
double min;

Kokkos::Min<double> min_reducer(min);
Kokkos::parallel_reduce( “MinReduce”, N, KOKKOS_LAMBDA (const int& x, double& lmin) {
  double val = (1.0*x- 7.2) * (1.0*x- 7.2) + 3.5;
  min_reducer.join(lmin, val); 
}, min_reducer);

printf(“Min: %lf\n”, min);
```