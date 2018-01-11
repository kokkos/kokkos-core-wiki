```cpp
Kokkos::RangePolicy<ExecSpace, WorkTag, Index>
```
Template parameters: an [Execution Space](API-Core#execution-spaces) `ExecSpace`, a work tag, and an integral `Index` type (e.g. `int`). Any subset of these parameters may be omitted, but those provided must be in the same order shown above. `ExecSpace` defaults to `Kokkos::DefaultExecutionSpace`, `WorkTag` defaults to `void`, and `Index` defaults to `ExecSpace::index_type`.

If the `WorkTag` is not `void`, then the user functor must take as its first argument an object of type `WorkTag`.  Otherwise, no work tag argument should be accepted. The next argument after the work tag should be an integer of type `Index`. For example:
```cpp
Kokkos::parallel_for(Kokkos::RangePolicy<int>, KOKKOS_LAMBDA(const int& i) {
  /* ... */
});
```
