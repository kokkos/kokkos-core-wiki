# `ParallelScanTag()`

Header File: `Kokkos_ExecPolicy.hpp`

A tag used in team size calculation functions to indicate that the functor for which a team size is being requested is being used in a [`parallel_scan`](../parallel-dispatch/parallel_scan)

Usage: 
```c++
using PolicyType = Kokkos::TeamPolicy<>; 
PolicyType policy;
int recommended_team_size = policy.team_size_recommended(
  Functor, Kokkos::ParallelScanTag());
```

## Synopsis 
```c++
struct ParallelScanTag{};
```

## Public Class Members

  None

### Typedefs
   
 None

### Constructors
 
 Default

### Functions

 None
