# Deprecation



## Kokkos-3.x


  |  **Deprecated Feature**    |  **Replacement**          | **Reason**              
  | -------------------------  | ------------------------  | ------------------------- 
  |                                             |                                      | 
  |  `Kokkos::is_reducer_type`                  |  `Kokkos::is_reducer`  |  Improve API
  |  Array reductions with raw pointer          |  Use `Kokkos::View` as return argument  |  Improve API
  |  `OffsetView` constructors taking `index_list_type`     |  `Kokkos::pair` (CPU and GPU)  |  Streamline arguments to `::pair` function
  |  Overloads of `Kokkos::sort` taking a parameter `bool always_use_kokkos_sort`  |  Use `Kokkos::BinSort` if required, or call `Kokkos::sort` without bool parameter  |  Updating overloads
  |                                             |                                                                  |
  |  **PUBLIC HEADERS UPDATES**                 |                                                                  |
  |  Guard against non-public header inclusion  |  **Core PUBLIC HEADERS**:  `Kokkos_Core.hpp`,                    |  Improve API
  |                                             |                            `Kokkos_Macros.hpp`,                  |
  |                                             |                            `Kokkos_Atomic.hpp`,                  |
  |                                             |                            `Kokkos_DetectionIdiom.hpp`,          |
  |                                             |                            `Kokkos_MathematicalConstants.hpp`,   |
  |                                             |                            `Kokkos_MathematicalFunctions.hpp`,   |
  |                                             |                            `Kokkos_NumericTraits.hpp`,           |
  |                                             |                            `Kokkos_Array.hpp`,                   |
  |                                             |                            `Kokkos_Complex.hpp`,                 |
  |                                             |                            `Kokkos_Pair.hpp`,                    |
  |                                             |                            `Kokkos_Half.hpp`,                    |
  |                                             |                            `Kokkos_Timer.hpp`                    |
  |                                             |                                                                  |
  |  Guard against non-public header inclusion  |  **Algorithms PUBLIC HEADERS**:  `Kokkos_StdAlgorithms.hpp`,       |  Improve API
  |                                             |                                  `Kokkos_Random.hpp`,              |
  |                                             |                                  `Kokkos_Sort.hpp`                 |
  |                                             |                                                                    |
  |  Guard against non-public header inclusion  |  **Containers PUBLIC HEADERS**:  `Kokkos_Bitset.hpp`,              | Improve API
  |                                             |                                  `Kokkos_DualView.hpp`,            |
  |                                             |                                  `Kokkos_DynRankView.hpp`,         |
  |                                             |                                  `Kokkos_ErrorReporter.hpp`,       |
  |                                             |                                  `Kokkos_Functional.hpp`,          |
  |                                             |                                  `Kokkos_OffsetView.hpp`,          |
  |                                             |                                  `Kokkos_ScatterView.hpp`,         |
  |                                             |                                  `Kokkos_StaticCrsGraph.hpp`,      |
  |                                             |                                  `Kokkos_UnorderedMap.hpp`,        |
  |                                             |                                  `Kokkos_Vector.hpp`               |
  |                                             |                                                                    |
  |  `Kokkos_UniqueToken.hpp` not a public header                     |  `Kokkos_Core.hpp`                           |
  |  `Kokkos_Threads.hpp` not a public header                         |  `Kokkos_Core.hpp`                           |  Improve API
  |  `Kokkos_Serial.hpp` not a public header                          |  `Kokkos_Core.hpp`                           |  Improve API
  |  `Kokkos_AnonymousSpace.hpp` not a public header                  |  `Kokkos_Core.hpp`                           |  Improve API
  |  `Kokkos_Atomics_Desul_Config.hpp` not a public header            |  `Kokkos_Atomic.hpp`                         |  Improve API
  |  `Kokkos_Vectorization.hpp` not a public header                   |  `Kokkos_Core.hpp`                           |  Improve API
  |  `Kokkos_OpenACC.hpp` not a public header                         |  `Kokkos_Core.hpp`                           |  Improve API
  |  `Kokkos_OpenACCSpace.hpp` not a public header                    |  `Kokkos_Core.hpp`                           |  Improve API
  |  `Kokkos_MasterLock.hpp` not a public header                      |  `Kokkos_Core.hpp`                           |  Improve API
  |  `Kokkos_View.hpp` not a public header                            |  `Kokkos_Core.hpp`                           |  Improve API
  |  `Kokkos_ExecPolicy.hpp` not a public header                      |  `Kokkos_Core.hpp`                           |  Improve API
  |  `Kokkos_Future.hpp` not a public header                          |  `Kokkos_Core.hpp`                           |  Improve API
  |  `Kokkos_GraphNode.hpp` not a public header                       |  `Kokkos_Core.hpp`                           |  Improve API
  |  `Kokkos_HBWSpace.hpp` not a public header                        |  `Kokkos_Core.hpp`                           |  Improve API
  |  `Kokkos_ScratchSpace.hpp` not a public header                    |  `Kokkos_Core.hpp`                           |  Improve API
  |  `Kokkos_Crs.hpp` not a public header                             |  `Kokkos_Core.hpp`                           |  Improve API
  |  `Kokkos_SYCL_Space.hpp` not a public header                      |  `Kokkos_Core.hpp`                           |  Improve API
  |  `Kokkos_SYCL.hpp` not a public header                            |  `Kokkos_Core.hpp`                           |  Improve API
  |  `Kokkos_Cuda.hpp` not a public header                            |  `Kokkos_Core.hpp`                           |  Improve API
  |  `Kokkos_CudaSpace.hpp` not a public header                       |  `Kokkos_Core.hpp`                           |  Improve API
  |  `KokkosExp_MDRangePolicy.hpp` not a public header                |  `Kokkos_Core.hpp`                           |  Improve API
  |  `Kokkos_Tuners.hpp` not a public header                          |  `Kokkos_Core.hpp`                           |  Improve API
  |  `Kokkos_HIP_Space.hpp` not a public header                       |  `Kokkos_Core.hpp`                           |  Improve API
  |  `Kokkos_HIP.hpp` not a public header                             |  `Kokkos_Core.hpp`                           |  Improve API
  |  `Kokkos_Rank.hpp` not a public header                            |  `Kokkos_Core.hpp`                           |  Improve API
  |  `Kokkos_Atomics_Desul_Volatile_Wrapper.hpp` not a public header  |  `Kokkos_Core.hpp`                           |  Improve API
  |  `Kokkos_Atomics_Desul_Wrapper.hpp` not a public header           |  `Kokkos_Core.hpp`                           |  Improve API
  |  `Kokkos_MinMaxClamp.hpp` not a public header                     |  `Kokkos_Core.hpp`                           |  Improve API
  |  `Kokkos_Concepts.hpp` not a public header                        |  `Kokkos_Core.hpp`                           |  Improve API
  |  `Kokkos_MemoryPool.hpp` not a public header                      |  `Kokkos_Core.hpp`                           |  Improve API
  |  `Kokkos_Parallel_Reduce.hpp` not a public header                 |  `Kokkos_Core.hpp`                           |  Improve API
  |  `Kokkos_TaskScheduler.hpp` not a public header                   |  `Kokkos_Core.hpp`                           |  Improve API
  |  `Kokkos_TaskScheduler_fwd.hpp` not a public header               |  `Kokkos_Core.hpp`                           |  Improve API
  |  `Kokkos_hwloc.hpp` not a public header                           |  `Kokkos_Core.hpp`                           |  Improve API
  |  `Kokkos_PointerOwnership.hpp` not a public header                |  `Kokkos_Core.hpp`                           |  Improve API
  |  `Kokkos_OpenMPTarget.hpp` not a public header                    |  `Kokkos_Core.hpp`                           |  Improve API
  |  `Kokkos_OpenMPTargetSpace.hpp` not a public header               |  `Kokkos_Core.hpp`                           |  Improve API
  |  `Kokkos_Layout.hpp` not a public header                          |  `Kokkos_Core.hpp`                           |  Improve API
  |  `Kokkos_MemoryTraits.hpp` not a public header                    |  `Kokkos_Core.hpp`                           |  Improve API
  |  `Kokkos_LogicalSpaces.hpp` not a public header                   |  `Kokkos_Core.hpp`                           |  Improve API
  |  `Kokkos_Extents.hpp` not a public header                         |  `Kokkos_Core.hpp`                           |  Improve API
  |  `Kokkos_WorkGraph.hpp` not a public header                       |  `Kokkos_Core.hpp`                           |  Improve API
  |                                                                                     |                            |
  |  Raise deprecation warnings if non-empty WorkTag class is used                      |  Use empty WorkTag class   |  Improve API
  |  `: secName(sectionName)` in `class ProfilingSection`                               |  Remove constructor        |  Improve API
  |  `KOKKOS_DEPRECATED std::string getName() { return secName; }`                      |  Remove function           |  Improve API
  |  `KOKKOS_DEPRECATED uint32_t getSectionID() { return secID; }`                      |  Remove function           |  Improve API
  |  `const std::string secName;`                                                       |  Remove variable           |  Improve API
  |  `using ActiveExecutionMemorySpace KOKKOS_DEPRECATED = Kokkos::HostSpace;`          |  Remove type alias         |  Improve API
  |  `using ActiveExecutionMemorySpace KOKKOS_DEPRECATED = void;`                       |  Remove type alias         |  Improve API
  |  `using is_array_layout KOKKOS_DEPRECATED = Kokkos::is_array_layout<T>;`            |  Remove type alias         |  Improve API 
  |  `using is_execution_policy KOKKOS_DEPRECATED = Kokkos::is_execution_policy<T>;`    |  Remove type alias         |  Improve API
  |  `using is_execution_space KOKKOS_DEPRECATED = Kokkos::is_execution_space<T>;`      |  Remove type alias         |  Improve API
  |  `using is_memory_space KOKKOS_DEPRECATED = Kokkos::is_memory_space<T>;`            |  Remove type alias         |  Improve API
  |  `using is_memory_traits KOKKOS_DEPRECATED = Kokkos::is_memory_traits<T>;`          |  Remove type alias         |  Improve API
  |  `using host_memory_space KOKKOS_DEPRECATED = do_not_use_host_memory_space;`        |  Remove type alias         |  Improve API
  |  `using host_execution_space KOKKOS_DEPRECATED = do_not_use_host_execution_space;`  |  Remove type alias         |  Improve API           
  |                                                                                     |                            |
  |  using host_mirror_space KOKKOS_DEPRECATED = std::conditional_t<                    |  Remove type alias         |  Improve API
  |  std::is_same<execution_space, do_not_use_host_execution_space>::value &&           |                            |
  |      std::is_same<memory_space, do_not_use_host_memory_space>::value,               |                            |
  |  T,                                                                                 |                            |
  |  Kokkos::Device<do_not_use_host_execution_space,                                    |                            |
  |                  do_not_use_host_memory_space>>;                                    |                            |
  |                                                                                     |                            |
  |  `namespace Impl`                                                                   |  Remove `namespace Impl`   |  Improve API
  |                                                                                     |                            |
  |  template <typename T>                                                              |                            |
  |  using is_space KOKKOS_DEPRECATED = Kokkos::is_space<T>;                            |  Remove type alias         |  Improve API
  |                                                                                     |                            |
  |  template <typename AccessSpace, typename MemorySpace>                              |                            |
  |  using SpaceAccessibility KOKKOS_DEPRECATED =                                       |  Remove type alias         |  Improve API
  |  Kokkos::SpaceAccessibility<AccessSpace, MemorySpace>;                              |                            |  Improve API
  |                                                                                     |                            |
  |  #define KOKKOS_RESTRICT_EXECUTION_TO_DATA(DATA_SPACE, DATA_PTR)\                   |  Remove macro              |  Improve API 
  |  Kokkos::Impl::verify_space<Kokkos::Impl::ActiveExecutionMemorySpace,               |                            |
  |                                ```DATA_SPACE>::check();                             |                            |
  |                                                                                     |                            |
  |  #define KOKKOS_RESTRICT_EXECUTION_TO_(DATA_SPACE)                                  |  Remove macro              |  Improve API
  |    Kokkos::Impl::verify_space<Kokkos::Impl::ActiveExecutionMemorySpace,             |                            |
  |                               ```DATA_SPACE>::check();                              |                            |
  |
  |
  |  `parallel_*` overloads taking the label as trailing argument                 |  `Kokkos::parallel_*("KokkosViewLabel", policy, f);`      |  Consistent ordering of parameters
  |  Embedded types (`argument_type`, `first_argument_type`, and `second_argument_type`) in `std::function`  |  Use `decltype` (if required)  |  Align `Kokkos_Functional.hpp` with deprecation in `std::function` (`#include <functional>`)
  |  `InitArguments` struct | `InitializationSettings()` object with query-able attributes                           |  Transparent and understandable initialization
  |  `finalize_all()`                                                                   |  `finalize()`              |  Improve  API
  |                                                                                     |                                                               |
  |  **COMMAND LINE ARGUMENTS UPDATES**                                                 |                                                               |
  |  Command-line arguments (other than `--help`) not prefixed with `kokkos-*`          | **UPDATED COMMAND-LINE ARGUMENTS**:  `--kokkos-num-threads`,  |  Improve API
  |                                                                                     |                                      `--kokkos-device-id`,    |
  |                                                                                     |                                      `--kokkos-num-devices`   |
  |                                                                                     |                                                               |
  |  `--[kokkos-]numa` command-line argument and `KOKKOS_NUMA` environment variable     |  `--kokkos-num-threads`   |  Harmonize option nomenclature with that of C++ `std::thread` library
  |  `--[kokkos-]threads` command-line argument  |  `--kokkos-num-threads`                                          |  Improve API
  |
  |  Warn about `parallel_reduce` cases that call `join()` with arguments qualified by `volatile` keyword           |  Remove `volatile` overloads      |  Streamline API
  |  `static void partition_master(F const& f, int requested_num_partitions = 0, int requested_partition_size = 0)` |  Remove function                  |  Improve API
  |  `void OpenMPInternal::validate_partition_impl(const int nthreads, int &num_partitions, int &partition_size)`   |  Remove function                  |  Improve API
  |  `std::vector<OpenMP> OpenMP::partition(...) { return std::vector<OpenMP>(1); }`                                |  Remove function                  |  Improve API
  |  `OpenMP OpenMP::create_instance(...) { return OpenMP(); }`                                                     |  Remove function                  |  Improve API
  |  `static void validate_partition(const int nthreads, int& num_partitions, int& partition_size)`                 |  Remove function                  |  Improve API
  |  `!std::is_void<T>::value &&`  |  `std::is_empty<T>::value &&` (C++17)                                                                              |  Improve API
  |  `static void validate_partition_impl(const int nthreads, int& num_partitions, int& partition_size)`            |  Remove function                  |  Improve API
  |  `void OpenMP::partition_master(F const& f, int num_partitions, int partition_size)                             |  Remove function                  |  Improve API
  |  `class MasterLock<OpenMP>`                                                                                     |  Remove class                     |  Improve API
  |  `class KOKKOS_ATTRIBUTE_NODISCARD ScopeGuard`                                                                  |  Remove class                     |  Improve API
  |  `create_mirror_view` taking `WithOutInitializing` as first argument | `create_mirror_view(Kokkos::Impl::WithoutInitializing_t wi, Kokkos::View<T, P...> const& v)`  |  Improve API
  |  `!std::is_empty<typename base_t::work_tag>::value && !std::is_void<typename base_t::work_tag>::value`          |  Remove condition                 |  Improve API
  |  `partition(...)`, `partition_master` for HPX backend                                                           |  Remove function                  |  Improve API
  |  `constexpr`                                                                                                    |  Remove specifier                 |  Improve API
  |  `#define KOKKOS_THREAD_LOCAL` macro                                                                            |  `thread_local`                   |  Improve API
  |  `vector_length() const`                                                                                        |  Remove function                  |  Improve API
  |  `class MasterLock`                                                                                             |  Remove class                     |  Improve API
  |  `Kokkos::Impl::is_view`                                                                                        | `Kokkos::is_view`                 |  Improve API
  |  `inline int vector_length() const`                                                                             |  Remove function                  |  Improve API  
  |                                                                                                                 |                                   |
  |  **CUDA DEPRECATION**                                                                                           |                                   |
  |                                                                                                                 |                                   |
  |  `void CudaSpace::access_error()`                                                                               |  Remove function                  |  Improve API
  |  `int CudaUVMSpace::number_of_allocations()`                                                                    |  Remove function                  |  Improve API
  |  `inline void cuda_internal_safe_call_deprecated()`                                                             | `#define CUDA_SAFE_CALL(call)`    |  Improve API
  |                                                                                                                 |                                   |
  |                                                                                                                 |                                   |
  |  `KOKKOS_DEPRECATED static void access_error();`                                                                |  Remove function                  |  Improve API  
  |  `KOKKOS_DEPRECATED static void access_error(const void* const);`                                               |  Remove function                  |
  |                                                                                                                 |                                   |
  |                                                                                                                 |                                   |
  |  `KOKKOS_DEPRECATED static int number_of_allocations();`                                                        |  Remove function                  |  Improve API 
  |                                                                                                                 |                                   |
  |  `using ActiveExecutionMemorySpace KOKKOS_DEPRECATED = Kokkos::CudaSpace;`                                      |  Remove type alias                |  Improve API
  |                                                                                                                 |                                   |
  |                                                                                                                 |                                   |
  |  **HIP DEPRECATION**                                                                                            |                                   |
  |                                                                                                                 |                                   |
  |  `using ActiveExecutionMemorySpace KOKKOS_DEPRECATED = Kokkos::Experimental::HIPSpace;`                         |  Remove type alias                |  Improve API
  |  `KOKKOS_DEPRECATED void Experimental::HIPSpace::access_error()`                                                |  Remove function                  |  Improve API
  |  `KOKKOS_DEPRECATED void Experimental::HIPSpace::access_error(const void* const)`                               |  Remove function                  |  Improve API
  |  `KOKKOS_DEPRECATED int vector_length() const { return impl_vector_length();`                                   |  Remove function                  |
  |                                                                                                                 |                                   |
  |  inline void hip_internal_safe_call_deprecated(hipError_t e, const char* name,                                  |  Remove function                  |  Improve API
  |                                                const char* file = nullptr,                                      |                                   |
  |                                                const int line   = 0)                                            |                                   |
  |                                                                                                                 |                                   |
  |  #define HIP_SAFE_CALL(call)                                                                                    |  Remove macro                     |  Improve API
  |    Kokkos::Impl::hip_internal_safe_call_deprecated(call, #call, __FILE__, __LINE__)                             |                                   |
  |                                                                                                                 |                                   |
  |                                                                                                                 |                                   |
  |  **SYCL DEPRECATION**                                                                                           |                                   |
  |                                                                                                                 |
  |  `using ActiveExecutionMemorySpace KOKKOS_DEPRECATED = Kokkos::Experimental::SYCLDeviceUSMSpace;`               |  Remove type alias                |  Improve API
  |                                                                                                                 |                                   |
  |                                                                                                                 |                                   |
  |  **PROMOTION FROM EXPERIMENTAL TO KOKKOKS NAMESPACE**                                                           |                                   |
  |                                                                                                                 |                                   |
  |  `Kokkos::Experimental::aMathFunction`                                                                          |  Use `namespace Kokkos`           |  Promote to Kokkos namespace
  |                                                                                                                 |                                   |
  |  namespace Experimental {                                                                                       |  Use `namespace Kokkos`           |  Promote to Kokkos namespace
  |  using ::Kokkos::clamp;                                                                                         |                                   |
  |  using ::Kokkos::max;                                                                                           |                                   |
  |  using ::Kokkos::min;                                                                                           |                                   |
  |  using ::Kokkos::minmax;                                                                                        |                                   |
  |  }  // namespace Experimental                                                                                   |                                   |
  |                                                                                                                 |                                   |
  |   namespace Kokkos {                                                                                            |  Use `namespace Kokkos`           |  Promote to Kokkos namespace
  |   namespace Experimental {                                                                                      |                                   |
  |   using Iterate KOKKOS_DEPRECATED = Kokkos::Iterate;                                                            |                                   |
  |   template <typename... Properties>                                                                             |                                   |
  |   using MDRangePolicy KOKKOS_DEPRECATED = Kokkos::MDRangePolicy<Properties...>;                                 |                                   |
  |   template <unsigned N, Kokkos::Iterate OuterDir = Kokkos::Iterate::Default,                                    |                                   |
  |             Kokkos::Iterate InnerDir = Kokkos::Iterate::Default>                                                |                                   |
  |   using Rank KOKKOS_DEPRECATED = Kokkos::Rank<N, OuterDir, InnerDir>;                                           |                                   |
  |  }  // namespace Experimental                                                                                   |                                   |
  |  }  // namespace Kokkos                                                                                         |                                   |
  |                                                                                                                 |                                   |
  |                                                                                                                 |                                   |
  |                                                                                                                 |                                   |
  |  **UNIT TEST DEPRECATION**                                                                                      |                                   |
  |                                                                                                                 |                                   |
  |  double *sums_ptr = sums;                                                                                       |  Remove test                      |  Update testing
  |  parallel_reduce(range, functor, sums_ptr);                                                                     |                                   |
  |  ASSERT_EQ(sums[0], 6 * N0 * N1);                                                                               |                                   |
  |  ASSERT_EQ(sums[1], 3 * N0 * N1);                                                                               |                                   |
  |                                                                                                                 |                                   |
  |  void take_initialization_settings(Kokkos::InitializationSettings const&) {}                                    |  Remove test                      |  Update testing
  |  TEST(defaultdevicetype,                                                                                        |                                   |
  |       init_arguments_implicit_conversion_to_initialization_settings) {                                          |                                   |
  |    Kokkos::InitArguments arguments;                                                                             |                                   |
  |    take_initialization_settings(arguments);  // check that conversion is implicit                               |                                   |
  |    arguments.device_id      = 1;                                                                                |                                   |
  |    arguments.tune_internals = true;                                                                             |                                   |
  |    Kokkos::InitializationSettings settings{arguments};                                                          |                                   |
  |    EXPECT_FALSE(settings.has_num_threads());                                                                    |                                   |
  |    EXPECT_TRUE(settings.has_device_id());                                                                       |                                   |
  |    EXPECT_EQ(settings.get_device_id(), 1);                                                                      |                                   |
  |    EXPECT_FALSE(settings.has_num_devices());                                                                    |                                   |
  |    EXPECT_FALSE(settings.has_skip_device());                                                                    |                                   |
  |    EXPECT_FALSE(settings.has_disable_warnings());                                                               |                                   |
  |    EXPECT_TRUE(settings.has_tune_internals());                                                                  |                                   |
  |    EXPECT_TRUE(settings.get_tune_internals());                                                                  |                                   |
  |    EXPECT_FALSE(settings.has_tools_help());                                                                     |                                   |
  |    EXPECT_FALSE(settings.has_tools_libs());                                                                     |                                   |
  |    EXPECT_FALSE(settings.has_tools_args());}                                                                    |                                   |
  |                                                                                                                 |                                   |
  |  Test result in host pointer in `parallel_reduce` (`ASSERT_EQ(host_result(j), (ScalarType)correct);`            |  Remove test case                 |  Update testing
  |                                                                                                                 |                                   |
  |  Kokkos::parallel_reduce(                                                                                       |  Remove test case                 |  Update testing
  |      policy, ReducerWithJoinThatTakesVolatileQualifiedArgs{}, result);                                          |                                   |
  |  ASSERT_EQ(result.err, no_error);                                                                               |                                   |
  |                                                                                                                 |                                   |
  |  `TEST(openmp, partition_master)`                                                                               |  Remove test                      |  Update testing
  |                                                                                                                 |                                   |
  |                                                                                                                 |                                   |


