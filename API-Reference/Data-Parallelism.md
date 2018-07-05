## Parallel patterns

Parallel execution patterns for composing algorithms.

|Function  |Description                  |
|:---------|:----------------------------|
|[parallel_for](Kokkos%3A%3Aparallel_for) | Executes user code in parallel |
|[parallel_reduce](Kokkos%3A%3Aparallel_reduce)  | Executes user code to perform a reduction in parallel |
|[parallel_scan](Kokkos%3A%3Aparallel_scan)  | Executes user code to generate a prefix sum in parallel |

## Built-in Reducers

Reducer objects used in conjunction with [parallel_reduce](Kokkos%3A%3Aparallel_reduce).

|Reducer  |Description                  |
|:---------|:----------------------------|
|[BAnd](Kokkos%3A%3ABAnd) | Binary 'And' reduction |
|[BOr](Kokkos%3A%3ABOr) | Binary 'Or' reduction |
|[LAnd](Kokkos%3A%3ALAnd) | Logical 'And' reduction |
|[LOr](Kokkos%3A%3ALOr) | Logical 'Or' reduction |
|[Max](Kokkos%3A%3AMax) | Maximum reduction |
|[Max](Kokkos%3A%3AMaxLoc) | Reduction providing maximum and an associated index |
|[Min](Kokkos%3A%3AMin) | Minimum reduction |
|[Min](Kokkos%3A%3AMinLoc) | Reduction providing minimum and an associated index |
|[MinMax](Kokkos%3A%3AMinMax) | Reduction providing both minimum and maximum |
|[MinMax](Kokkos%3A%3AMinMaxLoc) | Reduction providing both minimum and maximum and associated indicies |
|[Prod](Kokkos%3A%3AProd) | Multiplicative reduction |
|[Sum](Kokkos%3A%3ASum) | Sum reduction |

