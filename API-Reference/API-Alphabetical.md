All functions and classes listed here are part of the `Kokkos::` namespace. 

|Name |Library | Category | Description                  |
|:---------|:--------|:-----------|:----------------------------|
| | [Algorithm](API-Container) | | Algorithm description |
|Copy_Functor| Algorithm | Sort | View Copy |
|Copy_Permute_Functor| Algorithm | Sort | View Copy |
|Rand| Algorithm | Random Number | Generator Type (12), draw options (3) |
|Rand| Algorithm | Random Number | Generator Type (12), draw options (3) |
|Random_XorShift64_Pool| Algorithm | Random Number | Random Number Generator, pool for threads  |
|Random_XorShift64| Algorithm | Random Number | Random Number Generator for 12 types, plus normal distribution|
|init| Algorithm | Random Number | initialize state using seed for Random_XorShift64_Pool |
|Random_XorShift1024_Pool| Algorithm | Random Number | Random Number Generator, 1024 bit, pool for threads  |
|Random_XorShift64| Algorithm | Random Number | Random Number Generator for Cuda Device (12 types, normal distribution)|
|Random_XorShift1024| Algorithm | Random Number | Random Number Generator for Cuda Device (12 types, normal distribution)|
|Random_XorShift64| Algorithm | Random Number | Random Number Generator for ROCm Device (12 types, normal distribution)|
|Random_XorShift1024| Algorithm | Random Number | Random Number Generator for ROCm Device (12 types, normal distribution)|| | | | |
|fill_random| Algorithm | Random Number | create sample space to fit a (0 to) range or begin-end space |
| | [Container](API-Container) | | Container description |
| | | | |
|[BAnd](Kokkos%3A%3ABAnd) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism) | Reducer for Binary 'And' reduction |
|[BOr](Kokkos%3A%3ABOr) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism) | Reducer for Binary 'Or' reduction |
|[complex](Kokkos%3A%3AComplex) | [Core](API-Core) | [Utilities](API-Utilities) | Complex numbers which work on host and device |
|[LAnd](Kokkos%3A%3ALAnd) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism) | Reducer for Logical 'And' reduction |
|[LOr](Kokkos%3A%3ALOr) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism) | Reducer for Logical 'Or' reduction |
|[Max](Kokkos%3A%3AMax) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism) | Reducer for Maximum reduction |
|[MaxLoc](Kokkos%3A%3AMaxLoc) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism) | Reducer for Reduction providing maximum and an associated index |
|[MDRangePolicy](Kokkos%3A%3ARangePolicy) | [Core](API-Core) | [Execution Policies](Execution-Policies)| Policy to iterate over a multidimensional index range. |
|[Min](Kokkos%3A%3AMin) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism) | Reducer for Minimum reduction |
|[MinLoc](Kokkos%3A%3AMinLoc) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism) | Reducer for Reduction providing minimum and an associated index |
|[MinMax](Kokkos%3A%3AMinMax) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism) | Reducer for Reduction providing both minimum and maximum |
|[MinMaxLoc](Kokkos%3A%3AMinMaxLoc) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism) | Reducer for Reduction providing both minimum and maximum and associated indicies |
|[parallel_for](Kokkos%3A%3Aparallel_for) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism)| Bulk execute of independent work items. |
|[parallel_reduce](Kokkos%3A%3Aparallel_reduce) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism)| Bulk execute of independent work items, which contribute to a reduction. |
|[parallel_scan](Kokkos%3A%3Aparallel_scan) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism)| Bulk execute of work items, which a simple pre- or postfix scan dependency. |
|[PerTeam](Kokkos%3A%3ANestedPolicies) | [Core](API-Core) | [Execution Policies](Execution-Policies)| Policy used in single construct to indicate once per team execution. |
|[PerThread](Kokkos%3A%3ANestedPolicies) | [Core](API-Core) | [Execution Policies](Execution-Policies)| Policy used in single construct to indicate once per thread execution. |
|[Prod](Kokkos%3A%3AProd) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism) | Reducer for Multiplicative reduction |
|[RangePolicy](Kokkos%3A%3ARangePolicy) | [Core](API-Core) | [Execution Policies](Execution-Policies)| Policy to iterate over a 1D index range. |
|[Sum](Kokkos%3A%3ASum) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism) | Reducer for Sum reduction |
|[TeamHandleConcept](Kokkos%3A%3ATeamHandleConcept) | [Core](API-Core) | [Execution Policies](Execution-Policies)| Provides the concept for the `member_type` of a [TeamPolicy](Kokkos%3A%3ATeamPolicy). |
|[TeamPolicy](Kokkos%3A%3ATeamPolicy) | [Core](API-Core) | [Execution Policies](Execution-Policies)| Policy to iterate over a 1D index range, assigning to each iteration a team of threads. |
|[TeamThreadRange](Kokkos%3A%3ANestedPolicies) | [Core](API-Core) | [Execution Policies](Execution-Policies)| Policy to iterate over a 1D index range with the threads of a team. |
|[ThreadVectorRange](Kokkos%3A%3ANestedPolicies) | [Core](API-Core) | [Execution Policies](Execution-Policies)| Policy to iterate over a 1D index range with the vector lanes of a thread. |
|[Timer](Kokkos%3A%3ATimer) | [Core](API-Core) | [Utilities](API-Utilities)| A basic timer returning seconds |
|[View](Kokkos%3A%3AView) | [Core](API-Core) | [View](Kokkos%3A%3AView)| A multi-dimensional array |
|DualView | [Core](API-Core) | View| A multi-dimensional array on Host and Device |
|Subview | [Core](API-Core) | View| A multi-dimensional array which is a slice of a view |
