# `Kokkos::fence`

Header File: `Kokkos_Core.hpp`

In general, kernels in Kokkos are running asynchronously and launching them return immediately.
This implies that its effects might not be visible to the calling thread.
Kokkos::fence() guarantees that all parallel patterns in all execution spaces are guaranteed to have completed upon return
and that any side-effects are visible.
Effectively, calling Kokkos::fence() is equivalent to calling ExecutionSpace::fence() for all enabled execution spaces.

Usage:
timing kernels
```cpp
Kokkos::Timer timer;
// some kernel call, like Kokkos::parallel_for(...);
Kokkos::fence();
double time = timer.seconds();
```
using UVM
```cpp
Kokkos::View<int*, Kokkos::CudaUVM> view("uvm_view", 10);
Kokkos::parallel_for(10, KOKKOS_LAMBDA (int n) { view(n) = n; });
Kokkos::fence();
for (unsigned int i=0; i<view.size(); ++i)
  std::cout << view(i) << std::endl;
```

## Synopsis 

```cpp
void fence();
```
~        
