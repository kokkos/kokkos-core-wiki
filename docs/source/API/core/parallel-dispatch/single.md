# `fence()`

Header File: `Kokkos_Core.hpp`

Usage:

```c++
In a parallel dispatch statement, execute a lambda once per Thread or Team.
- Kokkos::single` accepts two policies:
  - Kokkos::PerTeam restricts execution of the lambda's body to once per team
  - Kokkos::PerThread restricts execution of the lambda body to once per thread (i.e., to only one vector lane in a thread).

```

The `single` function can take two arguments:  an execution `Policy` and a `Lambda`.  `Kokkos::single` is only valid for the `Lambda` to capture variables by value (i.e., [=], and not [&]). That `Lambda` takes zero arguments or one argument by reference.  If `Lambda` takes one argument, the final value of that argument is broadcast to every executor, for example, every vector lane of the Thread, or every Thread (and vector lane) of the Team. 


## Interface

```c++
Kokkos::single(Policy,Lambda);
```

### Parameters

- `Policy`: Defines how the execution will be performed (in the context of a parallel dispatch statement)..

### Requirements

- `Kokkos::single(Policy,Lambda)` can be called inside an existing parallel region (i.e. inside the `operator()` of a functor or `Lambda`).

## Semantics
- Focuses execution in parallel regions to a single work item for the `Policy`.


## Examples

```
using Kokkos::parallel_for;
using Kokkos::parallel_reduce;
using Kokkos::TeamThreadRange;
using Kokkos::ThreadVectorRange;
using Kokkos::PerThread;

TeamPolicy<...> policy (...);
typedef TeamPolicy<...>::member_type team_member;

parallel_for (policy, KOKKOS_LAMBDA (const team_member& thread) {
 // ...

  parallel_for (TeamThreadRange (thread, 100),
    KOKKOS_LAMBDA (const int& i) {
      double sum = 0;
      // Perform a vector reduction with a thread
      parallel_reduce (ThreadVectorRange (thread, 100),
        [=] (int i, double& lsum) {
          // ...
          lsum += ...;
      }, sum);
      // Add the result value into a team shared array.
      // Make sure it is only added once per thread.
      Kokkos::single (PerThread (), [=] () {
          shared_array(i) += sum;
      });
  });

  double sum;
  parallel_reduce (TeamThreadRange (thread, 99),
    KOKKOS_LAMBDA (int i, double& lsum) {
      // Add the result value into a team shared array.
      // Make sure it is only added once per thread.
      Kokkos::single (PerThread (thread), [=] () {
          lsum += someFunction (shared_array(i),
                                shared_array(i+1));
      });
  }, sum);

  // Add the per team contribution to global memory.
  Kokkos::single (PerTeam (thread), [=] () {
    global_array(thread.league_rank()) = sum;
  });
});

```
