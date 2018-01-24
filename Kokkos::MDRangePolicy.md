```cpp
Kokkos::MDRangePolicy<ExecSpace, Rank<int, IterationPatternOuter, IterationPatternInner>, Schedule, WorkTag, Index>
```
Template parameters:  
  &nbsp;&nbsp;an [Execution Space](Execution-Space-API) `ExecSpace`,  
  &nbsp;&nbsp;a `Rank` type templated on  
  &nbsp;&nbsp;&nbsp;&nbsp;  (1) the rank (an integral type),  
  &nbsp;&nbsp;&nbsp;&nbsp;  (2) the iteration pattern for traversing between tiles (options: `Iterate::Left`, `Iterate::Right`, `Iterate::Default`),  
  &nbsp;&nbsp;&nbsp;&nbsp;  (3) the iteration pattern for traversing between tiles (options: `Iterate::Left`, `Iterate::Right`, `Iterate::Default`),  
  &nbsp;&nbsp;a `Schedule` type (options: `Dynamic`, `Static` (Default)),  
  &nbsp;&nbsp;a work tag,  
  &nbsp;&nbsp;an integral `Index` type (e.g. `int`).  

`Rank<int>` is required; any other subset of these may be omitted, but those provided must be in the same order shown above. Two optional IterationPattern template parameters to `Rank` default to `Kokkos::Iterate::Default` which depend on the `ExecSpace`  
`ExecSpace` defaults to `Kokkos::DefaultExecutionSpace`  
`Schedule` defaults to `Kokkos::Static`  
`WorkTag` defaults to `void`  
`Index` defaults to `ExecSpace::index_type`.  

If the `WorkTag` is not `void`, then the user functor must take as its first argument an object of type `WorkTag`.  Otherwise, no work tag argument should be accepted. The next argument after the work tag should be an integer of type `Index`. For example:
```cpp
Kokkos::parallel_for(Kokkos::MDRangePolicy< Kokkos::Rank<2, Kokkos::IterateRight, Kokkos::IterateRight>, int>( {{range_begin_0, range_begin_1}}, {{range_end_0, range_end_1}} ), KOKKOS_LAMBDA(const int& i, const int& j) {
  /* ... */
});
```
