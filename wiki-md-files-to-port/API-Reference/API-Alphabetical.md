All functions and classes listed here are part of the `Kokkos::` namespace. 

### Algorithms
|Name |Library | Category | Description                  |
|:---------|:--------|:-----------|:----------------------------|
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
|[OffsetView](Offset-View) | [Containers](API-Containers) | View | View structure supporting non-zero start indicies. |
|[ScatterView](Kokkos%3A%3AScatterView) | [Containers](API-Containers) | View | View structure to transpartently support atomic and data replication strategies for scatter-reduce algorithms. |
|[StaticCrsGraph](Kokkos%3A%3AStaticCrsGraph) | [Containers](API-Containers) | View | A non-resizable CRS graph structure with view semantics. |
|[UnorderedMap](Unordered-Map) | [Containers](API-Containers) | View | A map data structure optimized for concurrent inserts. |
|[vector](Kokkos%3A%3Avector) | [Containers](API-Containers) | View | A class providing similar interfaces to `std::vector`. |


### Core

|Name |Library | Category | Description                  |
|:---------|:--------|:-----------|:----------------------------|
|[abort](Kokkos%3A%3Aabort) | [Core](API-Core) | [Utilities](API-Utilities) | Causes abnormal program termination. | 
|[atomic_exchange](Kokkos%3A%3Aatomic_exchange) | [Core](API-Core) | [Atomic-Operations](Atomic-Operations) | Atomic operation which exchanges a value and returns the old. | 
|[atomic_compare_exchange](Kokkos%3A%3Aatomic_compare_exchange) | [Core](API-Core) | [Atomic-Operations](Atomic-Operations) | Atomic operation which exchanges a value only if the old value matches a comparison value and returns the old value. | 
|[atomic_compare_exchange_strong](Kokkos%3A%3Aatomic_compare_exchange_strong) | [Core](API-Core) | [Atomic-Operations](Atomic-Operations) | Atomic operation which exchanges a value only if the old value matches a comparison value and returns true if the exchange is executed. | 
|[atomic_load](Kokkos%3A%3Aatomic_load) | [Core](API-Core) | [Atomic-Operations](Atomic-Operations) | Atomic operation which loads a value. | 
|[atomic_\[op\]](Kokkos%3A%3Aatomic_op) | [Core](API-Core) | [Atomic-Operations](Atomic-Operations) | Atomic operation which don't return anything. | 
|[atomic_fetch_\[op\]](Kokkos%3A%3Aatomic_fetch_op) | [Core](API-Core) | [Atomic-Operations](Atomic-Operations) | Various atomic operations which return the old value. | 
|[atomic_\[op\]_fetch](Kokkos%3A%3Aatomic_op_fetch) | [Core](API-Core) | [Atomic-Operations](Atomic-Operations) | Various atomic operations which return the updated value. | 
|[atomic_store](Kokkos%3A%3Aatomic_store) | [Core](API-Core) | [Atomic-Operations](Atomic-Operations) | Atomic operation which stores a value. | 
|[BAnd](Kokkos%3A%3ABAnd) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism) | Reducer for Binary 'And' reduction |
|[BOr](Kokkos%3A%3ABOr) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism) | Reducer for Binary 'Or' reduction |
|[complex](Kokkos%3A%3AComplex) | [Core](API-Core) | [STL Compatibility](STL-Compatibility) | Complex numbers which work on host and device |
|[(X)create_mirror](Kokkos%3A%3Acreate_mirror) | [Core](API-Core) | [View](API-View) | Mirror Host data to Device data |
|[(X)create_mirror_view](Kokkos%3A%3Acreate_mirror) | [Core](API-Core) | [View](API-View) | Mirror Host data to Device data |
|[Cuda](Kokkos%3A%3ACuda) | [Core](API-Core) | [Spaces](API-Spaces) | The CUDA Execution Space. |
|[CudaSpace](Kokkos%3A%3ACudaSpace) | [Core](API-Core) | [Spaces](API-Spaces) | The primary CUDA Memory Space. |
|[CudaUVMSpace](Kokkos%3A%3ACudaUVMSpace) | [Core](API-Core) | [Spaces](API-Spaces) | The CUDA Memory Space providing access to unified memory page migratable allocations. |
|[CudaHostPinnedSpace](Kokkos%3A%3ACudaHostPinnedSpace) | [Core](API-Core) | [Spaces](API-Spaces) | The CUDA Memrory Space providing access to host pinned GPU-accessible host memory. |
|[deep_copy](Kokkos%3A%3Adeep_copy) | [Core](API-Core) | [View](API-View) | Copy Views |
|[ExecutionPolicy Concept](ExecutionPolicyConcept) | [Core](API-Core) | [Execution Policies](Execution-Policies) | Concept for execution policies. |
|[ExecutionSpace concept](Kokkos%3A%3AExecutionSpaceConcept) | [Core](API-Core) | [Spaces](API-Spaces) | Concept for execution spaces. |
|[fence](Kokkos%3A%3Afence) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism) | Fences execution spaces. |
|[finalize](Kokkos%3A%3Afinalize) | [Core](API-Core) | [Initialization and Finalization](initialize-and-finalize) | function to finalize Kokkos |
|[HostSpace](Kokkos%3A%3AHostSpace) | [Core](API-Core) | [Spaces](API-Spaces) | The primary Host Memory Space. |
|[HPX](Kokkos%3A%3AHPX) | [Core](API-Core) | [Spaces](API-Spaces) | Execution space using the HPX runtime system execution mechanisms. |
|[initialize](Kokkos%3A%3Ainitialize) | [Core](API-Core) | [Initialization and Finalization](initialize-and-finalize) | function to initialize Kokkos |
|[is_array_layout](Kokkos%3A%3Ais_array_layout) | [Core](API-Core) | | Trait to detect types that model the [Layout concept](Kokkos%3A%3ALayoutConcept) |
|[is_execution_policy](Kokkos%3A%3Ais_execution_policy) | [Core](API-Core) | | Trait to detect types that model [ExecutionPolicy concept](Kokkos%3A%3AExecutionPolicyConcept) |
|[is_execution_space](Kokkos%3A%3Ais_execution_space) | [Core](API-Core) | | Trait to detect types that model [ExecutionSpace concept](Kokkos%3A%3AExecutionSpaceConcept) |
|[is_memory_space](Kokkos%3A%3Ais_memory_space) | [Core](API-Core) | | Trait to detect types that model [MemorySpace concept](Kokkos%3A%3AMemorySpaceConcept) |
|[is_memory_traits](Kokkos%3A%3Ais_memory_traits) | [Core](API-Core) | | Trait to detect specializations of [Kokkos::MemoryTraits](Kokkos%3A%3AMemoryTraits) |
|[is_reducer](Kokkos%3A%3Ais_reducer) | [Core](API-Core) | | Trait to detect types that model the [Reducer concept](Kokkos%3A%3AReducerConcept) |
|[is_space](Kokkos%3A%3Ais_space) | [Core](API-Core) | | Trait to detect types that model the [Space concept](Kokkos%3A%3ASpaceConcept) |
|[LayoutLeft](Kokkos%3A%3ALayoutLeft) | [Core](API-Core) | [Views](Views) | Memory Layout matching Fortran |
|[LayoutRight](Kokkos%3A%3ALayoutRight) | [Core](API-Core) | [Views](Views) | Memory Layout matching C |
|[LayoutStride](Kokkos%3A%3ALayoutStride) | [Core](API-Core) | [Views](Views) | Memory Layout for arbitrary strides |
|[kokkos_free](Kokkos%3A%3Akokkos_free) | [Core](API-Core) | [Spaces](API-Spaces) | Dellocates previously allocated memory |
|[kokkos_malloc](Kokkos%3A%3Akokkos_malloc) | [Core](API-Core) | [Spaces](API-Spaces) | Allocates memory |
|[kokkos_realloc](Kokkos%3A%3Akokkos_realloc) | [Core](API-Core) | [Spaces](API-Spaces) | Expands previously allocated memory block |
|[LAnd](Kokkos%3A%3ALAnd) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism) | Reducer for Logical 'And' reduction |
|[LOr](Kokkos%3A%3ALOr) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism) | Reducer for Logical 'Or' reduction |
|[Max](Kokkos%3A%3AMax) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism) | Reducer for Maximum reduction |
|[MaxLoc](Kokkos%3A%3AMaxLoc) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism) | Reducer for Reduction providing maximum and an associated index |
|[(U)MDRangePolicy](Kokkos%3A%3AMDRangePolicy) | [Core](API-Core) | [Execution Policies](Execution-Policies)| Policy to iterate over a multidimensional index range. |
|[MemorySpace concept](Kokkos%3A%3AMemorySpaceConcept) | [Core](API-Core) | [Spaces](API-Spaces) | Concept for execution spaces. |
|[Min](Kokkos%3A%3AMin) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism) | Reducer for Minimum reduction |
|[MinLoc](Kokkos%3A%3AMinLoc) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism) | Reducer for Reduction providing minimum and an associated index |
|[MinMax](Kokkos%3A%3AMinMax) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism) | Reducer for Reduction providing both minimum and maximum |
|[MinMaxLoc](Kokkos%3A%3AMinMaxLoc) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism) | Reducer for Reduction providing both minimum and maximum and associated indicies |
|[OpenMP](Kokkos%3A%3AOpenMP) | [Core](API-Core) | [Spaces](API-Spaces) | Execution space using non-target OpenMP parallel execution mechanisms. |
|[OpenMPTarget](Kokkos%3A%3AOpenMPTarget) | [Core](API-Core) | [Spaces](API-Spaces) | Execution space using targetoffload OpenMP parallel execution mechanisms. |
|[pair](Kokkos%3A%3Apair) | [Core](API-Core) | [STL Compatibility](STL-Compatibility)| Device compatible std::pair analogue
|[parallel_for](Kokkos%3A%3Aparallel_for) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism)| Bulk execute of independent work items. |
|[ParallelForTag](Kokkos%3A%3AParallelForTag) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism)| Tag passed to team\_size functions
|[parallel_reduce](Kokkos%3A%3Aparallel_reduce) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism)| Bulk execute of independent work items, which contribute to a reduction. |
|[ParallelReduceTag](Kokkos%3A%3AParallelReduceTag) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism)| Tag passed to team\_size functions
|[parallel_scan](Kokkos%3A%3Aparallel_scan) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism)| Bulk execute of work items, which a simple pre- or postfix scan dependency. |
|[ParallelScanTag](Kokkos%3A%3AParallelScanTag) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism)| Tag passed to team\_size functions
|[PerTeam](Kokkos%3A%3ANestedPolicies) | [Core](API-Core) | [Execution Policies](Execution-Policies)| Policy used in single construct to indicate once per team execution. |
|[PerThread](Kokkos%3A%3ANestedPolicies) | [Core](API-Core) | [Execution Policies](Execution-Policies)| Policy used in single construct to indicate once per thread execution. |
|[Prod](Kokkos%3A%3AProd) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism) | Reducer for Multiplicative reduction |
|[RangePolicy](Kokkos%3A%3ARangePolicy) | [Core](API-Core) | [Execution Policies](Execution-Policies)| Policy to iterate over a 1D index range. |
|[realloc](Kokkos%3A%3Arealloc) | [Core](API-Core) | [View](API-View) | Resize an existing view without maintaining the content |
|[ReducerConcept](Kokkos%3A%3AReducerConcept) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism)| Provides the concept for Reducers. |
|[resize](Kokkos%3A%3Aresize) | [Core](API-Core) | [View](API-View) | Resize an existing view while maintaining the content |
|[ScopeGuard](Kokkos%3A%3AScopeGuard) | [Core](API-Core) | [Initialization and Finalization](initialize-and-finalize) | class to aggregate initializing and finalizing Kokkos |
|[SpaceAccessibility](Kokkos%3A%3ASpaceAccessibility) | [Core](API-Core) | [Spaces](API-Spaces) | Facility to query accessibility rules between execution and memory spaces. |
|[subview](Kokkos%3A%3Asubview) | [Core](API-Core) | [View](API-View) | Crating multi-dimensional array which is a slice of a view |
|[Sum](Kokkos%3A%3ASum) | [Core](API-Core) | [Data-Parallelism](Data-Parallelism) | Reducer for Sum reduction |
|[TeamHandle concept](TeamHandleConcept) | [Core](API-Core) | [Execution Policies](Execution-Policies)| Provides the concept for the `member_type` of a [TeamPolicy](Kokkos%3A%3ATeamPolicy). |
|[(U)TeamPolicy](Kokkos%3A%3ATeamPolicy) | [Core](API-Core) | [Execution Policies](Execution-Policies)| Policy to iterate over a 1D index range, assigning to each iteration a team of threads. |
|[TeamThreadRange](Kokkos%3A%3ATeamThreadRange) | [Core](API-Core) | [Execution Policies](Execution-Policies)| Policy to iterate over a 1D index range with the threads of a team. |
|[TeamVectorRange](Kokkos%3A%3ATeamVectorRange) | [Core](API-Core) | [Execution Policies](Execution-Policies)| Policy to iterate over a 1D index range with the threads and vector lanes of a team. |
|[ThreadVectorRange](Kokkos%3A%3AThreadVectorRange) | [Core](API-Core) | [Execution Policies](Execution-Policies)| Policy to iterate over a 1D index range with the vector lanes of a thread. |
|[Timer](Kokkos%3A%3ATimer) | [Core](API-Core) | [Utilities](API-Utilities)| A basic timer returning seconds |
|[View](Kokkos%3A%3AView) | [Core](API-Core) | [View](API-View)| A multi-dimensional array |
|[View-like Type Concept](ViewLike) | [Core](API-Core) | [View](API-View) | A set of class templates that act like a View |
