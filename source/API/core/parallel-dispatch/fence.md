# `fence()`

Header File: `Kokkos_Core.hpp`

Usage:

```c++
Kokkos::fence();
```

Blocks on completion of all outstanding asynchronous Kokkos operations.
That includes parallel dispatch (e.g. [parallel_for()](parallel_for), [parallel_reduce()](parallel_reduce) 
and [parallel_scan()](parallel_scan)) as well as asynchronous data operations such as three-argument [deep_copy](../view/deep_copy).

Note: there is a execution space instance specific `fence` too: [ExecutionSpaceConcept](ExecutionSpaceConcept)

## Interface

```c++
void Kokkos::fence();
```

```c++
void Kokkos::fence(const std::string& label);
```

### Parameters

- `label`: A label to identify a specific fence in fence profiling operations. `label` does not have to be unique.

### Requirements

- `Kokkos::fence()` cannot be called inside an existing parallel region (i.e. inside the `operator()` of a functor or lambda).

## Semantics

- Blocks on completion of all outstanding asynchronous works. Side effects of outstanding work will be observable upon completion of the `fence` call - that means `Kokkos::fence()` implies a memory fence.

## Examples

### Timing kernels
```c++
Kokkos::Timer timer;
// This operation is asynchronous, without a fence 
// one would time only the launch overhead
Kokkos::parallel_for("Test", N, functor);
Kokkos::fence();
double time = timer.seconds();
```

### Use with asynchronous deep copy

```c++
Kokkos::deep_copy(exec1, a,b);
Kokkos::deep_copy(exec2, a,b);
// do some stuff which doesn't touch a or b
Kokkos::parallel_for("Test", N, functor);

// wait for all three operations to finish
Kokkos::fence();

// do something with a and b
```
