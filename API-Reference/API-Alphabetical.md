All functions and classes listed here are part of the `Kokkos::` namespace. 

### Algorithms
|Name |Library | Category | Description                  |
|:---------|:--------|:-----------|:----------------------------|
|Copy_Functor| Algorithm | Sort | View Copy |
|Copy_Permute_Functor| Algorithm | Sort | View Copy |
|Rand| Algorithm | Random Number | Generator Type (12), draw options (3) |
|Rand| Algorithm | Random Number | Generator Type (12), draw options (3) |
|Random_XorShift64_Pool| Algorithm | Random Number | Random Number Generator, pool for threads  |
|Random_XorShift64| Algorithm | Random Number | Random Number Generator for 12 types, plus normal distribution|
|init| Algorithm | Random Number | initialize state using seed for Random_XorShift64_Pool |
|Random_XorShift1024_Pool| Algorithm | Random Number | Random Number Generator, 1024 bit, pool for threads  |
|Random_XorShift1024| Algorithm | Random Number | Random Number Generator for 12 types, plus normal distribution)|
|fill_random| Algorithm | Random Number | create sample space to fit a (0 to) range or begin-end space |


### Containers
|Name |Library | Category | Description                  |
|:---------|:--------|:-----------|:----------------------------|
|[Bitset](Kokkos%3A%3ABitset) | [Containers](API-Containers) | View | A concurrent Bitset class. |
|[DualView](Kokkos%3A%3ADualView) | [Containers](API-Containers) | View | Host-Device Mirror of View with Host-Device Memory |
|[DynRankView](Kokkos%3A%3ADynRankView) | [Containers](API-Containers) | View | A view which can determine its rank at runtime. |
|[DynamicView](Kokkos%3A%3ADynamicView) | [Containers](API-Containers) | View | A view which can change its size dynamically. |
|[ErrorReporter](Kokkos%3A%3AErrorReporter) | [Containers](API-Containers) | View | A class supporting error recording in parallel code. |
|[OffsetView](Kokkos%3A%3AOffsetView) | [Containers](API-Containers) | View | View structure supporting non-zero start indicies. |
|[ScatterView](Kokkos%3A%3AScatterView) | [Containers](API-Containers) | View | View structure to transpartently support atomic and data replication strategies for scatter-reduce algorithms. |
|[StaticCrsGraph](Kokkos%3A%3AStaticCrsGraph) | [Containers](API-Containers) | View | A non-resizable CRS graph structure with view semantics. |
|[UnorderedMap](Kokkos%3A%3AUnorderedMap) | [Containers](API-Containers) | View | A map data structure optimized for concurrent inserts. |
|[vector](Kokkos%3A%3Avector) | [Containers](API-Containers) | View | A class providing similar interfaces to `std::vector`. |


### Core

|Name |Library | Category | Description                  |
|:---------|:--------|:-----------|:----------------------------|
|[(X)atomic_exchange](Kokkos%3A%3Aatomic_exchange) | [Core](API-Core) | [Atomic-Operations](Atomic-Operations) | Atomic operation which exchanges a value and returns the old. | 
|[(X)atomic_compare_exchange](Kokkos%3A%3Aatomic_compare_exchange) | [Core](API-Core) | [Atomic-Operations](Atomic-Operations) | Atomic operation which exchanges a value only if the old value matches a comparison value and returns the old value. | 
|[(X)atomic_compare_exchange_strong](Kokkos%3A%3Aatomic_compare_exchange_strong) | [Core](API-Core) | [Atomic-Operations](Atomic-Operations) | Atomic operation which exchanges a value only if the old value matches a comparison value and returns true if the exchange is executed. | 
|[(X)atomic_\[op\]](Kokkos%3A%3Aatomic_op) | [Core](API-Core) | [Atomic-Operations](Atomic-Operations) | Atomic operation which don't return anything. | 
|[(X)atomic_fetch_\[op\]](Kokkos%3A%3Aatomic_fetch_op) | [Core](API-Core) | [Atomic-Operations](Atomic-Operations) | Various atomic operations which return the old value. | 
|[(X)atomic_\[op\]_fetch](Kokkos%3A%3Aatomic_op_fetch) | [Core](API-Core) | [Atomic-Operations](Atomic-Operations) | Various atomic operations which return the updated value. | 
|[BAnd](Kokkos%3A%3ABAnd) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism) | Reducer for Binary 'And' reduction |
|[BOr](Kokkos%3A%3ABOr) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism) | Reducer for Binary 'Or' reduction |
|[(U)complex](Kokkos%3A%3AComplex) | [Core](API-Core) | [Utilities](API-Utilities) | Complex numbers which work on host and device |
|[(X)create_mirror](Kokkos%3A%3Acreate_mirror) | [Core](API-Core) | [View](Data-Parallelism) | Mirror Host data to Device data |
|[(X)create_mirror_view](Kokkos%3A%3Acreate_mirror_view) | [Core](API-Core) | [View](Data-Parallelism) | Mirror Host data to Device data |
|[Cuda](Kokkos%3A%3ACuda) | [Core](API-Core) | [Spaces](Spaces) | The CUDA Execution Space. |
|[CudaSpace](Kokkos%3A%3ACudaSpace) | [Core](API-Core) | [Spaces](Spaces) | The primary CUDA Memory Space. |
|[CudaUVMSpace](Kokkos%3A%3ACudaUVMSpace) | [Core](API-Core) | [Spaces](Spaces) | The CUDA Memory Space providing access to unified memory page migratable allocations. |
|[CudaHostPinnedSpace](Kokkos%3A%3ACudaHostPinnedSpace) | [Core](API-Core) | [Spaces](Spaces) | The CUDA Memrory Space providing access to host pinned GPU-accessible host memory. |
|[ExecutionSpace Concept](ExecutionSpaceConcept) | [Core](API-Core) | [Spaces](Spaces) | Concept for execution spaces. |
|[HostSpace](Kokkos%3A%3AHostSpace) | [Core](API-Core) | [Spaces](Spaces) | The primary Host Memory Space. |
|[HPX](Kokkos%3A%3AHPX) | [Core](API-Core) | [Spaces](Spaces) | Execution space using the HPX runtime system execution mechanisms. |
|[LAnd](Kokkos%3A%3ALAnd) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism) | Reducer for Logical 'And' reduction |
|[LOr](Kokkos%3A%3ALOr) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism) | Reducer for Logical 'Or' reduction |
|[Max](Kokkos%3A%3AMax) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism) | Reducer for Maximum reduction |
|[MaxLoc](Kokkos%3A%3AMaxLoc) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism) | Reducer for Reduction providing maximum and an associated index |
|[(U)MDRangePolicy](Kokkos%3A%3AMDRangePolicy) | [Core](API-Core) | [Execution Policies](Execution-Policies)| Policy to iterate over a multidimensional index range. |
|[MemorySpaceConcept](MemorySpaceConcept) | [Core](API-Core) | [Spaces](Spaces) | Concept for execution spaces. |
|[OpenMP](Kokkos%3A%3AOpenMP) | [Core](API-Core) | [Spaces](Spaces) | Execution space using non-target OpenMP parallel execution mechanisms. |
|[OpenMPTarget](Kokkos%3A%3AOpenMPTarget) | [Core](API-Core) | [Spaces](Spaces) | Execution space using targetoffload OpenMP parallel execution mechanisms. |
|[Min](Kokkos%3A%3AMin) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism) | Reducer for Minimum reduction |
|[MinLoc](Kokkos%3A%3AMinLoc) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism) | Reducer for Reduction providing minimum and an associated index |
|[MinMax](Kokkos%3A%3AMinMax) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism) | Reducer for Reduction providing both minimum and maximum |
|[MinMaxLoc](Kokkos%3A%3AMinMaxLoc) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism) | Reducer for Reduction providing both minimum and maximum and associated indicies |
|[parallel_for](Kokkos%3A%3Aparallel_for) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism)| Bulk execute of independent work items. |
|[parallel_reduce](Kokkos%3A%3Aparallel_reduce) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism)| Bulk execute of independent work items, which contribute to a reduction. |
|[(U)parallel_scan](Kokkos%3A%3Aparallel_scan) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism)| Bulk execute of work items, which a simple pre- or postfix scan dependency. |
|[PerTeam](Kokkos%3A%3ANestedPolicies) | [Core](API-Core) | [Execution Policies](Execution-Policies)| Policy used in single construct to indicate once per team execution. |
|[PerThread](Kokkos%3A%3ANestedPolicies) | [Core](API-Core) | [Execution Policies](Execution-Policies)| Policy used in single construct to indicate once per thread execution. |
|[Prod](Kokkos%3A%3AProd) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism) | Reducer for Multiplicative reduction |
|[RangePolicy](Kokkos%3A%3ARangePolicy) | [Core](API-Core) | [Execution Policies](Execution-Policies)| Policy to iterate over a 1D index range. |
|[(X)realloc](Kokkos%3A%3Arealloc) | [Core](API-Core) | View | Resize an existing view without maintaining the content |
|[(U)ReducerConcept](Kokkos%3A%3AReducerConcept) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism)| Provides the concept for Reducers. |
|[(X)resize](Kokkos%3A%3Aresize) | [Core](API-Core) | View | Resize an existing view while maintaining the content |
|[SpaceAccessibility](Kokkos%3A%3ASpaceAccessibility) | [Core](API-Core) | [Spaces](Spaces) | Facility to query accessibility rules between execution and memory spaces. |
|[(X)subview](Kokkos%3A%3Asubview) | [Core](API-Core) | View | Crating multi-dimensional array which is a slice of a view |
|[Sum](Kokkos%3A%3ASum) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism) | Reducer for Sum reduction |
|[TeamHandleConcept](Kokkos%3A%3ATeamHandleConcept) | [Core](API-Core) | [Execution Policies](Execution-Policies)| Provides the concept for the `member_type` of a [TeamPolicy](Kokkos%3A%3ATeamPolicy). |
|[(U)TeamPolicy](Kokkos%3A%3ATeamPolicy) | [Core](API-Core) | [Execution Policies](Execution-Policies)| Policy to iterate over a 1D index range, assigning to each iteration a team of threads. |
|[(U)TeamThreadRange](Kokkos%3A%3ANestedPolicies) | [Core](API-Core) | [Execution Policies](Execution-Policies)| Policy to iterate over a 1D index range with the threads of a team. |
|[(U)TeamVectorRange](Kokkos%3A%3ANestedPolicies) | [Core](API-Core) | [Execution Policies](Execution-Policies)| Policy to iterate over a 1D index range with the threads and vector lanes of a team. |
|[(U)ThreadVectorRange](Kokkos%3A%3ANestedPolicies) | [Core](API-Core) | [Execution Policies](Execution-Policies)| Policy to iterate over a 1D index range with the vector lanes of a thread. |
|[Timer](Kokkos%3A%3ATimer) | [Core](API-Core) | [Utilities](API-Utilities)| A basic timer returning seconds |
|[View](Kokkos%3A%3AView) | [Core](API-Core) | [View](Kokkos%3A%3AView)| A multi-dimensional array |
