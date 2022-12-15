# Deprecation

<!--- VERSION 3 DEPRECATION --->

## Kokkos-3.x

  |  **Deprecated**  |  **Replacement**  |  **Reason**                
  |  ------------  | ------------  |  ------------
  |  `Kokkos::is_reducer_type` |  `Kokkos::is_reducer`  |  Improve API
  |  Array reductions with raw pointer  |  Use `Kokkos::View` as return argument  |  Improve API
  |  `OffsetView` constructors taking `index_list_type`  |  `Kokkos::pair` (CPU and GPU)  |  Streamline arguments to `::pair` function
  |  Overloads of `Kokkos::sort` taking a parameter `bool always_use_kokkos_sort`  |  Use `Kokkos::BinSort` if required, or call `Kokkos::sort` without bool parameter  |  Updating overloads
  |  |  **PUBLIC HEADERS UPDATES** 
  |  Guard against non-public header inclusion  |  **Core PUBLIC HEADERS**:  |  Improve API
  |  `Kokkos_Core.hpp`,
  |  `Kokkos_Macros.hpp`,
  |  `Kokkos_Atomic.hpp`,
  |  `Kokkos_DetectionIdiom.hpp`,
  |  `Kokkos_MathematicalConstants.hpp`,
  |  `Kokkos_MathematicalFunctions.hpp`,
  |  `Kokkos_NumericTraits.hpp`,
  |  `Kokkos_Array.hpp`,
  |  `Kokkos_Complex.hpp`,
  |  `Kokkos_Pair.hpp`,
  |  `Kokkos_Half.hpp`,
  |  `Kokkos_Timer.hpp`
  |  Guard against non-public header inclusion  |  **Algorithms PUBLIC HEADERS**:  |  Improve API
  |  |  `Kokkos_StdAlgorithms.hpp`
  |  |  `Kokkos_Random.hpp`,
  |  |  `Kokkos_Sort.hpp`
  |  Guard against non-public header inclusion  |  **Containers PUBLIC HEADERS**:  | Improve API
  |  |  `Kokkos_Bitset.hpp`,
  |  |  `Kokkos_DualView.hpp`,
  |  |  `Kokkos_DynRankView.hpp`,
  |  |  `Kokkos_ErrorReporter.hpp`,
  |  |  `Kokkos_Functional.hpp`,
  |  |  `Kokkos_OffsetView.hpp`,
  |  |  `Kokkos_ScatterView.hpp`,
  |  |  `Kokkos_StaticCrsGraph.hpp`,
  |  |  `Kokkos_UnorderedMap.hpp`,
  |  |  `Kokkos_Vector.hpp`
  |  `Kokkos_UniqueToken.hpp` not a public header  |  `Kokkos_Core.hpp`  |  Improve API
  |  `Kokkos_Threads.hpp` not a public header  |  `Kokkos_Core.hpp`  |  Improve API
  |  `Kokkos_Serial.hpp` not a public header  |  `Kokkos_Core.hpp`  |  Improve API
  |  `Kokkos_AnonymousSpace.hpp` not a public header  |  `Kokkos_Core.hpp`  |  Improve API
  |  `Kokkos_Atomics_Desul_Config.hpp` not a public header  |  `Kokkos_Atomic.hpp`  |  Improve API
  |  `Kokkos_Vectorization.hpp` not a public header  |  `Kokkos_Core.hpp`  |  Improve API
  |  `Kokkos_OpenACC.hpp` not a public header  |  `Kokkos_Core.hpp`  |  Improve API
  |  `Kokkos_OpenACCSpace.hpp` not a public header  |  `Kokkos_Core.hpp`  |  Improve API
  |  `Kokkos_MasterLock.hpp` not a public header  |  `Kokkos_Core.hpp`  |  Improve API
  |  `Kokkos_View.hpp` not a public header  |  `Kokkos_Core.hpp`  |  Improve API
  |  `Kokkos_ExecPolicy.hpp` not a public header  |  `Kokkos_Core.hpp`  |  Improve API
  |  `Kokkos_Future.hpp` not a public header  |  `Kokkos_Core.hpp`  |  Improve API
  |  `Kokkos_GraphNode.hpp` not a public header  |  `Kokkos_Core.hpp`  |  Improve API
  |  `Kokkos_HBWSpace.hpp` not a public header  |  `Kokkos_Core.hpp`  |  Improve API
  |  `Kokkos_ScratchSpace.hpp` not a public header  |  `Kokkos_Core.hpp`  |  Improve API
  |  `Kokkos_Crs.hpp` not a public header  |  `Kokkos_Core.hpp`  |  Improve API
  |  `Kokkos_SYCL_Space.hpp` not a public header  |  `Kokkos_Core.hpp`  |  Improve API
  |  `Kokkos_SYCL.hpp` not a public header  |  `Kokkos_Core.hpp`  |  Improve API
  |  `Kokkos_Cuda.hpp` not a public header  |  `Kokkos_Core.hpp`  |  Improve API
  |  `Kokkos_CudaSpace.hpp` not a public header  |  `Kokkos_Core.hpp`  |  Improve API
  |  `KokkosExp_MDRangePolicy.hpp` not a public header  |  `Kokkos_Core.hpp`  |  Improve API
  |  `Kokkos_Tuners.hpp` not a public header  |  `Kokkos_Core.hpp`  |  Improve API
  |  `Kokkos_HIP_Space.hpp` not a public header  |  `Kokkos_Core.hpp`  |  Improve API
  |  `Kokkos_HIP.hpp` not a public header  |  `Kokkos_Core.hpp`  |  Improve API
  |  `Kokkos_Rank.hpp` not a public header  |  `Kokkos_Core.hpp`  |  Improve API
  |  `Kokkos_Atomics_Desul_Volatile_Wrapper.hpp` not a public header  |  `Kokkos_Core.hpp`  |  Improve API
  |  `Kokkos_Atomics_Desul_Wrapper.hpp` not a public header  |  `Kokkos_Core.hpp`  |  Improve API
  |  `Kokkos_MinMaxClamp.hpp` not a public header  |  `Kokkos_Core.hpp`  |  Improve API
  |  `Kokkos_Concepts.hpp` not a public header  |  `Kokkos_Core.hpp`  |  Improve API
  |  `Kokkos_MemoryPool.hpp` not a public header  |  `Kokkos_Core.hpp`  |  Improve API
  |  `Kokkos_Parallel_Reduce.hpp` not a public header  |  `Kokkos_Core.hpp`  |  Improve API
  |  `Kokkos_TaskScheduler.hpp` not a public header  |  `Kokkos_Core.hpp`  |  Improve API
  |  `Kokkos_TaskScheduler_fwd.hpp` not a public header  |  `Kokkos_Core.hpp`  |  Improve API
  |  `Kokkos_hwloc.hpp` not a public header  |  `Kokkos_Core.hpp`  |  Improve API
  |  `Kokkos_PointerOwnership.hpp` not a public header  |  `Kokkos_Core.hpp`  |  Improve API
  |  `Kokkos_OpenMPTarget.hpp` not a public header  |  `Kokkos_Core.hpp`  |  Improve API
  |  `Kokkos_OpenMPTargetSpace.hpp` not a public header  |  `Kokkos_Core.hpp`  |  Improve API
  |  `Kokkos_Layout.hpp` not a public header  |  `Kokkos_Core.hpp`  |  Improve API
  |  `Kokkos_MemoryTraits.hpp` not a public header  |  `Kokkos_Core.hpp`  |  Improve API
  |  `Kokkos_LogicalSpaces.hpp` not a public header  |  `Kokkos_Core.hpp`  |  Improve API
  |  `Kokkos_Extents.hpp` not a public header  |  `Kokkos_Core.hpp`  |  Improve API
  |  `Kokkos_WorkGraph.hpp` not a public header  |  `Kokkos_Core.hpp`  |  Improve API
  |  Raise deprecation warnings if non-empty WorkTag class is used  |  Use empty WorkTag class  |  Improve API
  |  `: secName(sectionName)` in `class ProfilingSection`  |  Remove constructor  |  Improve API
  |  `KOKKOS_DEPRECATED std::string getName() { return secName; }`  |  Remove function  |  Improve API
  |  `KOKKOS_DEPRECATED uint32_t getSectionID() { return secID; }`  |  Remove function           |  Improve API
  |  `const std::string secName;`  |  Remove variable  |  Improve API
  |  `using ActiveExecutionMemorySpace KOKKOS_DEPRECATED`  |  Remove type alias  |  Improve API
  |  `using ActiveExecutionMemorySpace KOKKOS_DEPRECATED`  |  Remove type alias  |  Improve API
  |  `using is_array_layout KOKKOS_DEPRECATED`  |  Remove type alias  |  Improve API 
  |  `using is_execution_policy KOKKOS_DEPRECATED`  |  Remove type alias  |  Improve API
  |  `using is_execution_space KOKKOS_DEPRECATED`  |  Remove type alias  |  Improve API
  |  `using is_memory_space KOKKOS_DEPRECATED`  |  Remove type alias  |  Improve API
  |  `using is_memory_traits KOKKOS_DEPRECATED`  |  Remove type alias  |  Improve API
  |  `using host_memory_space KOKKOS_DEPRECATED`  |  Remove type alias  |  Improve API
  |  `using host_execution_space KOKKOS_DEPRECATED`  |  Remove type alias  |  Improve API           
  |  `using host_mirror_space KOKKOS_DEPRECATED`  |  Remove type alias  |  Improve API
  |  `namespace Impl`  |  Remove `namespace Impl`  |  Improve API
  |  `using is_space KOKKOS_DEPRECATED`  |  Remove type alias  |  Improve API
  |  `using SpaceAccessibility KOKKOS_DEPRECATED`  |  Remove type alias  |  Improve API
  |  `#define KOKKOS_RESTRICT_EXECUTION_TO_DATA(DATA_SPACE, DATA_PTR)`  |  Remove macro  |  Improve API
  |  `#define KOKKOS_RESTRICT_EXECUTION_TO_(DATA_SPACE)`  |  Remove macro  |  Improve API
  |  `parallel_*` overloads taking the label as trailing argument  |  `Kokkos::parallel_*("KokkosViewLabel", policy, f);`  |  Consistent ordering of parameters
  |  Embedded types (`argument_type`, `first_argument_type`, and `second_argument_type`) in `std::function`  |  Use `decltype` (if required)  |  Align with deprecation in `std::function`
  |  `InitArguments` struct | `InitializationSettings()` class object with query-able attributes  |  Verifiable initialization
  |  `finalize_all()`  |  `finalize()`  |  Improve  API
  |  |  **COMMAND LINE ARGUMENTS UPDATES**
  |  Command-line arguments (other than `--help`) not prefixed with `kokkos-*`  | **UPDATED COMMAND-LINE ARGUMENTS**:  |  Improve API
  |  |  `--kokkos-num-threads`,
  |  |  `--kokkos-device-id`,
  |  |  `--kokkos-num-devices`
  |  `--[kokkos-]numa` command-line argument and `KOKKOS_NUMA` environment variable  |  `--kokkos-num-threads`  |  Align option nomenclature with `std::thread`
  |  `--[kokkos-]threads` command-line argument  |  `--kokkos-num-threads`  |  Improve API
  |  Warn about `parallel_reduce` cases that call `join()` with arguments qualified by `volatile` keyword  |  Remove `volatile` overloads  |  Streamline API
  |  `static void partition_master(F const& f, int requested_num_partitions = 0, int requested_partition_size = 0)`  |  Remove function  |  Improve API
  |  `void OpenMPInternal::validate_partition_impl(const int nthreads, int &num_partitions, int &partition_size)`  |  Remove function  |  Improve API
  |  `std::vector<OpenMP> OpenMP::partition(...) { return std::vector<OpenMP>(1); }`  |  Remove function  |  Improve API
  |  `OpenMP OpenMP::create_instance(...) { return OpenMP(); }`  |  Remove function  |  Improve API
  |  `static void validate_partition(const int nthreads, int& num_partitions, int& partition_size)`  |  Remove function  |  Improve API
  |  `!std::is_void<T>::value &&`  |  `std::is_empty<T>::value &&` (C++17)  |  Improve API
  |  `static void validate_partition_impl(const int nthreads, int& num_partitions, int& partition_size)`  |  Remove function  |  Improve API
  |  `void OpenMP::partition_master(F const& f, int num_partitions, int partition_size)`  |  Remove function  |  Improve API
  |  `class MasterLock<OpenMP>`  |  Remove class  |  Improve API
  |  `class KOKKOS_ATTRIBUTE_NODISCARD ScopeGuard`  |  Remove class  |  Improve API
  |  `create_mirror_view` taking `WithOutInitializing` as first argument | `create_mirror_view(Kokkos::Impl::WithoutInitializing_t wi, Kokkos::View<T, P...> const& v)`  |  Improve API
  |  `!std::is_empty<typename base_t::work_tag>::value && !std::is_void<typename base_t::work_tag>::value`  |  Remove condition  |  Improve API
  |  `partition(...)`, `partition_master` for HPX backend  |  Remove function  |  Improve API
  |  `constexpr`  |  Remove specifier  |  Improve API
  |  `#define KOKKOS_THREAD_LOCAL` macro  |  `thread_local`  |  Improve API
  |  `vector_length() const`  |  Remove function  |  Improve API
  |  `class MasterLock`  |  Remove class  |  Improve API
  |  `Kokkos::Impl::is_view`  |  `Kokkos::is_view`  |  Improve API
  |  `inline int vector_length() const`  |  Remove function  |  Improve API  
  |  |  **CUDA DEPRECATION** 
  |  `void CudaSpace::access_error()`  |  Remove function  |  Improve API
  |  `int CudaUVMSpace::number_of_allocations()` |  Remove function  |  Improve API
  |  `inline void cuda_internal_safe_call_deprecated()`  |  `#define CUDA_SAFE_CALL(call)`  |  Improve API
  |  `KOKKOS_DEPRECATED static void access_error();`  |  Remove function  |  Improve API  
  |  `KOKKOS_DEPRECATED static void access_error(const void* const);`  |  Remove function
  |  `KOKKOS_DEPRECATED static int number_of_allocations();`  |  Remove function  |  Improve API 
  |  `using ActiveExecutionMemorySpace KOKKOS_DEPRECATED`  |  Remove type alias  |  Improve API
  |  |  **HIP DEPRECATION**
  |  `using ActiveExecutionMemorySpace KOKKOS_DEPRECATED`  |  Remove type alias  |  Improve API
  |  `KOKKOS_DEPRECATED void Experimental::HIPSpace::access_error()`  |  Remove function  |  Improve API
  |  `KOKKOS_DEPRECATED void Experimental::HIPSpace::access_error(const void* const)`  |  Remove function  |  Improve API
  |  `KOKKOS_DEPRECATED int vector_length() const { return impl_vector_length();`  |  Remove function 
  |  `inline void hip_internal_safe_call_deprecated  |  Remove function  |  Improve API
  |  `#define HIP_SAFE_CALL(call)`  |  Remove macro  |  Improve API
  |  |**SYCL DEPRECATION**
  |  `using ActiveExecutionMemorySpace KOKKOS_DEPRECATED`  |  Remove type alias  |  Improve API
  |  |  **PROMOTION TO KOKKOKS NAMESPACE** 
  |  `Kokkos::Experimental::aMathFunction`  |  Use `namespace Kokkos`  |  Promote to Kokkos namespace
  |  `Kokkos::Experimental::clamp`  |  Use `namespace Kokkos`  |  Promote to Kokkos namespace
  |  `Kokkos::Experimental::max;`  |  Use `namespace Kokkos`  |  Promote to Kokkos namespace
  |  `Kokkos::Experimental::min;`  |  Use `namespace Kokkos`  |  Promote to Kokkos namespace
  |  `Kokkos::Experimental::minmax;`  |  Use `namespace Kokkos`  |  Promote to Kokkos namespace
  |  `using Iterate KOKKOS_DEPRECATED`  |  Remove type alias  |  Improve API
  |  `using MDRangePolicy KOKKOS_DEPRECATED`  |  Remove type alias  |  Improve API 
  |  `using Rank KOKKOS_DEPRECATED`  |  Remove type alias  |  Improve API  
  |  |  **UNIT TEST DEPRECATION**
  |  Test reduction of a pointer to a 1D array `parallel_reduce(range, functor, sums_ptr)`  |  Remove test  |  Update testing
  |  `void take_initialization_settings(Kokkos::InitializationSettings const&) {}`  |  Remove test  |  Update testing
  |  Test scalar result in host pointer in `parallel_reduce` `(ASSERT_EQ(host_result(j), (ScalarType)correct);`  |  Remove test case  |  Update testing
  |  Kokkos::parallel_reduce(policy, ReducerWithJoinThatTakesVolatileQualifiedArgs{}, result);  |  Remove test case  |  Update testing
  |  `TEST(openmp, partition_master)`  |  Remove test  |  Update testing
