```cpp
Kokkos::MDRangePolicy<ExecSpace, Rank<int, IterationPatternOuter, IterationPatternInner>, Schedule, WorkTag, Index>
```
Template parameters: an [Execution Space](Execution-Space-API) `ExecSpace`, a `Rank` type templated on (1) the rank (an integral type), (2) the iteration pattern for traversing between tiles (options: `Iterate::Left`, `Iterate::Right`, `Iterate::Default`), (3) the iteration pattern for traversing between tiles (options: `Iterate::Left`, `Iterate::Right`, `Iterate::Default`), a `Schedule` type (options: `Dynamic`, `Static` (Default)), a work tag, and an integral `Index` type (e.g. `int`). `Rank<int>` is required, any other subset of these may be omitted, but those provided must be in the same order shown above. `ExecSpace` defaults to `Kokkos::DefaultExecutionSpace`, IterationPatterns default to `Kokkos::Iterate::Default` which depend on the `ExecSpace`, `Schedule` defaults to `Kokkos::Static`, `WorkTag` defaults to `void`, and `Index` defaults to `ExecSpace::index_type`.

If the `WorkTag` is not `void`, then the user functor must take as its first argument an object of type `WorkTag`.  Otherwise, no work tag argument should be accepted. The next argument after the work tag should be an integer of type `Index`. For example:
```cpp
Kokkos::parallel_for(Kokkos::MDRangePolicy< Kokkos::Rank<2, Kokkos::IterateRight, Kokkos::IterateRight>, int>( {{range_begin_0, range_begin_1}}, {{range_end_0, range_end_1}} ), KOKKOS_LAMBDA(const int& i, const int& j) {
  /* ... */
});
```
