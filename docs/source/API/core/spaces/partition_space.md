# Kokkos::partition_space()

Header file: `Kokkos_Core.hpp`

Usage:

*Note: Currently `partition_space` is still in the namespace `Kokkos::Experimental`*

```c++
auto instances = Kokkos::partition_space(Kokkos::DefaultExecutionSpace(),1,1,1);
```

Creates new execution space instances which dispatch to the same underlying hardware resources as an existing execution space instance.
There is no implied synchronization relationship between the newly created instances and the pre-existing instance.

## Interface

```c++
template<class ExecSpace, class ... Args>
std::vector<ExecSpace> partition_space(const ExecSpace& space, Args...args);
```

### Parameters

- `space`: an execution space instance [ExecutionSpaceConcept](Kokkos%3A%3AExecutionSpaceConcept)
- `args`: the number of created instances is equal to `sizeof...(Args)`. The relative weight of `args` is a hint for the fraction of hardware resources of `space` to associate with each newly created instance.

### Requirements

- `(std::is_arithmetic_v<Args> && ...)` is `true`. 

### Semantics

- There is no implied synchronization relationship between any of the instances, specifically:
  - `instance[i]` is not fenced by `space.fence()`,
  - `instance[i]` is not fenced by `instance[j].fence()`, and
  - `space` is not fenced by `instance[i].fence()`.
- However, in practice these instances may block each other because they dispatch to the same hardware resources.
- The relative weight of `args` is used as a hint for the desired resource allocation. For example for a backend which uses discreet threads, weights of `{1,2}` would result in two instances where the first is associated with about 1/3rd of the threads of the original instance, and the second with 2/3rds. However, for some backends each returned instance may be a copy of the original one.
- *Note:* For `Cuda`, `HIP` and `SYCL` each newly created instance is associated with its own *stream*/*queue*.


## Examples

### Splitting an existing instance for use with concurrent kernels

```c++
template<class ExecSpace, class ... OtherParams>
void foo(const ExecSpace& space, OtherParams...params) {
  auto instances = Kokkos::partition_space(space,1,2);
  // dispatch two kernels, F1 needs less resources then F2
  // F1 and F2 may now execute concurrently
  Kokkos::parallel_for("F1",
    Kokkos::RangePolicy<ExecSpace>(instances[0],0,N1),
    Functor1(params...));
  Kokkos::parallel_for("F2",
    Kokkos::RangePolicy<ExecSpace>(instances[1],0,N2),
    Functor2(params...));

  // Wait for both
  // Note: space.fence() would NOT block execution of the instances
  instances[0].fence();
  instances[1].fence();
  Kokkos::parallel_for("F3",
    Kokkos::RangePolicy<ExecSpace>(space,0,N3),
    Functor3(params...));
}
```
