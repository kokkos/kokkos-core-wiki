# `Kokkos::RangePolicy`

Header File: `Kokkos_Core.hpp`

Usage: 
  ```c++
  Kokkos::RangePolicy<>(begin, end)
  Kokkos::RangePolicy<ARGS>(begin, end)
  Kokkos::RangePolicy<>(begin, end, args...)
  Kokkos::RangePolicy<ARGS>(begin, end, args...)
  Kokkos::RangePolicy<>(Space(), begin, end)
  Kokkos::RangePolicy<ARGS>(Space(), begin, end)
  Kokkos::RangePolicy<>(Space(), begin, end, args...)
  Kokkos::RangePolicy<ARGS>(Space(), begin, end, args...)
  ```

RangePolicy defines an execution policy for a 1D iteration space starting at begin and going to end with an open interval. 

# Interface 
  ```c++
  template<class ... Args>
  class Kokkos::RangePolicy;
  ```

## Parameters:

### Common Arguments for all Execution Policies

  * Execution Policies generally accept compile time arguments via template parameters and runtime parameters via constructor arguments or setter fucntions.
  * Template arguments can be given in arbitrary order.

| Argument | Options | Purpose |
| --- | --- | --- |
| ExecutionSpace | `Serial`, `OpenMP`, `Threads`, `Cuda`, `ROCm` | Specify the Execution Space to execute the kernel in. Defaults to `Kokkos::DefaultExecutionSpace`. |
| Schedule | `Schedule<Dynamic>`, `Schedule<Static>` | Specifiy scheduling policy for work items. `Dynamic` scheduling is implemented through a work stealing queue. Default is machine and backend specific. |
| IndexType | `IndexType<int>` | Specify integer type to be used for traversing the iteration space. Defaults to `int64_t`. |
| WorkTag | `SomeClass` | Specify the work tag type used to call the functor operator. Any arbitrary type defaults to `void`. |

### Requriements:


## Public Class Members

### Constructors
 
 * RangePolicy(): Default Constructor unitialized policy.
 * ```c++
   template<class ... InitArgs> 
   RangePolicy(const int64_t& begin, const int64_t& end, const InitArgs ... init_args)
   ```
   Provide a start and end index as well as optional arguments to control certain behavior (see below).
   
 * ```c++
   template<class ... InitArgs> 
   RangePolicy(const ExecutionSpace& space, const int64_t& begin, const int64_t& end, const InitArgs ... init_args)
   ```
   Provide a start and end index and an `ExecutionSpace` instance to use as the execution resource, as well as optional arguments to control certain behavior (see below).

#### Optional `InitArgs`:

 * `ChunkSize` : Provide a hint for optimal chunk-size to be used during scheduling.


## Examples

  ```c++
    RangePolicy<> policy_1(N);
    RangePolicy<Cuda> policy_2(5,N-5);
    RangePolicy<Schedule<Dynamic>, OpenMP> policy_3(n,m);
    RangePolicy<IndexType<int>, Schedule<Dynamic>> policy_4(K);
    RangePolicy<> policy_6(-3,N+3, ChunkSize(8));
    RangePolicy<OpenMP> policy_7(OpenMP(), 0, N, ChunkSize(4));
  ```

  Note: providing a single integer as a policy to a parallel pattern, implies a defaulted `RangePolicy`

  ```c++
    // These two calls are identical
    parallel_for("Loop", N, functor);
    parallel_for("Loop", RangePolicy<>(N), functor);
  ```


