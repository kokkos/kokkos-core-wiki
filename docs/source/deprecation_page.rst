Deprecation
-----------

.. IMPORTANT::
   Kokkos 4.0 requires C++17

Kokkos-3.x
~~~~~~~~~~

.. list-table::
   :widths: 40 40 20
   :header-rows: 1
   :align: left

   * - Heading Previous
     - Heading Replaced with
     - Heading Reason

   * - ``Kokkos::is_reducer_type``
     - ``Kokkos::is_reducer``
     - Improve API

   * - Array reductions with raw pointer
     - Use ``Kokkos::View`` as return argument
     - Improve API

   * - ``OffsetView`` constructors taking ``index_list_type``
     - ``Kokkos::pair`` (CPU and GPU)
     - Streamline arguments to ``::pair`` function

   * - Overloads of ``Kokkos::sort`` taking a parameter ``bool always_use_kokkos_sort``
     - Use ``Kokkos::BinSort`` if required, or call ``Kokkos::sort`` without bool parameter
     - Updating overloads
     
   * - Raise deprecation warnings if non-empty WorkTag class is used
     - Use empty WorkTag class
     - Improve API

  * - ``!std::is_empty<typename base_t::work_tag>::value && !std::is_void<typename base_t::work_tag>::value``
    - Remove condition
    - Improve API

   * - ``: secName(sectionName)`` in ``class ProfilingSection``
     - Remove constructor
     - Improve API
     
   * - ``std::string getName() { return secName; }``
     - Remove function
     - Improve API
     
   * - ``uint32_t getSectionID() { return secID; }``
     - Remove function
     - Improve API
 
   * - ``const std::string secName``
     - Remove variable
     - Improve API
     
   * - ``using ActiveExecutionMemorySpace``
     - Remove type alias
     - Improve API
     
   * - ``using ActiveExecutionMemorySpace``
     - Remove type alias
     - Improve API
     
   * - ``using is_array_layout``
     - Remove type alias
     - Improve API
     
   * - ``using is_execution_policy``
     - Remove type alias
     - Improve API
     
   * - ``using is_execution_space``
     - Remove type alias
     - Improve API

   * - ``using is_memory_space``
     - Remove type alias
     - Improve API

   * - ``using is_memory_traits``
     - Remove type alias
     - Improve API

   * - ``using host_memory_space``
     - Remove type alias
     - Improve API

   * - ``using host_execution_space``
     - Remove type alias
     - Improve API

   * - ``using host_mirror_space``
     - Remove type alias
     - Improve API

   * - ``namespace Impl``
     - Remove ``namespace Impl`
     - Improve API
     
   * - ``using is_space``
     - Remove type alias
     - Improve API
     
   * - ``using SpaceAccessibility``
     - Remove type alias
     - Improve API

   * - ``#define KOKKOS_RESTRICT_EXECUTION_TO_(DATA_SPACE)``
     - Remove macro
     - Improve API

  * - ``parallel_*`` overloads taking the label as trailing argument
    - ``Kokkos::parallel_*("KokkosViewLabel", policy, f);``
    - Consistent ordering of parameters
  
  * - ``InitArguments`` struct
    - ``InitializationSettings()`` class object with query-able attributes
    - Verifiable initialization
  
  * - ``finalize_all()``
    - ``finalize()``
    - Improve  API
  
  * - Warn about ``parallel_reduce`` cases that call ``join()`` with arguments qualified by ``volatile`` keyword
    - Remove ``volatile`` overloads
    - Streamline API
  
  * - ``static void partition_master(F const& f, int requested_num_partitions = 0, int requested_partition_size = 0)``
    - Remove function
    - Improve API
  
  * - ``std::vector<OpenMP> OpenMP::partition(...) { return std::vector<OpenMP>(1); }``
    - Remove function
    - Improve API
  
  * - ``OpenMP OpenMP::create_instance(...) { return OpenMP(); }``
    - Remove function
    - Improve API
  
  * - ``static void validate_partition(const int nthreads, int& num_partitions, int& partition_size)``
    - Remove function
    - Improve API
  
  * - ``void OpenMP::partition_master(F const& f, int num_partitions, int partition_size)``
    - Remove function
    - Improve API
  
  * - ``class MasterLock<OpenMP>``
    - Remove class
    - Improve API
  
  * - ``class KOKKOS_ATTRIBUTE_NODISCARD ScopeGuard``
    - Remove class
    - Improve API
  
  * - ``create_mirror_view`` taking ``WithOutInitializing`` as first argument
    - ``create_mirror_view(Kokkos::Impl::WithoutInitializing_t wi, Kokkos::View<T, P...> const& v)``
    - Improve API
  
  * - ``constexpr``
    - Remove specifier
    - Improve API
  
  * - ``#define KOKKOS_THREAD_LOCAL`` macro
    - ``thread_local``
    - Improve API
  
  * - ``vector_length() const``
    - Remove function
    - Improve API
  
  * - ``class MasterLock``
    - Remove class
    - Improve API
  
  * - ``Kokkos::Impl::is_view``
    - ``Kokkos::is_view``
    - Improve API
  
  * - ``inline int vector_length() const``
    - Remove function
    - Improve API
  
  * - Including private headers is deprecated
    - PUBLIC CORE HEADERS:  ``Kokkos_Core.hpp``, ``Kokkos_Macros.hpp``, ``Kokkos_Atomic.hpp``, ``Kokkos_DetectionIdiom.hpp``, ``Kokkos_MathematicalConstants.hpp``, ``Kokkos_MathematicalFunctions.hpp``, ``Kokkos_NumericTraits.hpp``, ``Kokkos_Array.hpp``, ``Kokkos_Complex.hpp``, ``Kokkos_Pair.hpp``, ``Kokkos_Half.hpp``, ``Kokkos_Timer.hpp``
    - Improve API
  
  * - Including private headers is deprecated
    - PUBLIC ALGORITHMS HEADERS:  ``Kokkos_StdAlgorithms.hpp``, ``Kokkos_Random.hpp``, ``Kokkos_Sort.hpp``
    - Improve API
  
  * - Including private headers is deprecated:  ``Kokkos_Bitset.hpp``, ``Kokkos_DualView.hpp``, ``Kokkos_DynRankView.hpp``, ``Kokkos_ErrorReporter.hpp``, ``Kokkos_Functional.hpp``, ``Kokkos_OffsetView.hpp``, ``Kokkos_ScatterView.hpp``, ``Kokkos_StaticCrsGraph.hpp``, ``Kokkos_UnorderedMap.hpp``, ``Kokkos_Vector.hpp``, ``Kokkos_UniqueToken.hpp``, ``Kokkos_Threads.hpp``, ``Kokkos_Serial.hpp``, ``Kokkos_AnonymousSpace.hpp``, ``Kokkos_Atomics_Desul_Config.hpp``, ``Kokkos_Vectorization.hpp``, ``Kokkos_OpenACC.hpp``, ``Kokkos_OpenACCSpace.hpp``, ``Kokkos_MasterLock.hpp``, ``Kokkos_View.hpp``, ``Kokkos_ExecPolicy.hpp``, ``Kokkos_Future.hpp``, ``Kokkos_GraphNode.hpp``, ``Kokkos_HBWSpace.hpp``, ``Kokkos_ScratchSpace.hpp``, ``Kokkos_Crs.hpp``, ``Kokkos_SYCL_Space.hpp``, ``Kokkos_SYCL.hpp``, ``Kokkos_Cuda.hpp``, ``Kokkos_CudaSpace.hpp``, ``KokkosExp_MDRangePolicy.hpp``, ``Kokkos_Tuners.hpp``, ``Kokkos_HIP_Space.hpp``, ``Kokkos_HIP.hpp``, ``Kokkos_Rank.hpp``, ``Kokkos_Atomics_Desul_Volatile_Wrapper.hpp``, ``Kokkos_Atomics_Desul_Wrapper.hpp``, ``Kokkos_MinMaxClamp.hpp``, ``Kokkos_Concepts.hpp``, ``Kokkos_MemoryPool.hpp``, ``Kokkos_Parallel_Reduce.hpp``, ``Kokkos_TaskScheduler.hpp``, ``Kokkos_TaskScheduler_fwd.hpp``, ``Kokkos_hwloc.hpp``, ``Kokkos_PointerOwnership.hpp``, ``Kokkos_OpenMPTarget.hpp``, ``Kokkos_OpenMPTargetSpace.hpp``, ``Kokkos_Layout.hpp``, ``Kokkos_MemoryTraits.hpp``, ``Kokkos_LogicalSpaces.hpp``, ``Kokkos_Extents.hpp``, ``Kokkos_AcquireUniqueTokenImpl.hpp``, ``Kokkos_CopyViews.hpp``, ``Kokkos_HostSpace.hpp``, ``Kokkos_HPX.hpp``, ``Kokkos_OpenMP.hpp``, ``Kokkos_Parallel.hpp``, ``Kokkos_WorkGraphPolicy.hpp``
  - PUBLIC HEADER:  ``Kokkos_Core.hpp``
  - Improve API
  
  * - Command-line arguments (other than ``--help``) not prefixed with ``kokkos-*``
    - ``--kokkos-num-threads``, ``--kokkos-device-id``, ``--kokkos-num-devices``, ``--kokkos-numa``, ``--kokkos-num-threads``, ``--kokkos-num-threads``
    - Improve API
  
  * - ``void CudaSpace::access_error()``
    - Remove function
    - Improve API
  
  * - ``int CudaUVMSpace::number_of_allocations()``
    - Remove function
    - Improve API
  
  * - ``inline void cuda_internal_safe_call_deprecated()``
    - ``#define CUDA_SAFE_CALL(call)``
    - Improve API
  
  * - ``partition(...)``, ``partition_master`` for HPX backend
    - Remove function 
    - Improve API

  * - ``static void access_error();``
    - Remove function
    - Improve API
  
  * - ``static void access_error(const void* const);``
    - Remove function
    - Improve API
  
  * - ``static int number_of_allocations();``
    - Remove function
    - Improve API
  
  * - ``using ActiveExecutionMemorySpace``
    - Remove type alias
    - Improve API
  
  * - ``using ActiveExecutionMemorySpace``
    - Remove type alias
    - Improve API
  
  * - ``void Experimental::HIPSpace::access_error()``
    - Remove function
    - Improve API
  
  * - ``void Experimental::HIPSpace::access_error(const void* const)``
    - Remove function
    - Improve API
  
  * - ``inline void hip_internal_safe_call_deprecated``
    - Remove function
    - Improve API
  
  * - ``#define HIP_SAFE_CALL(call)``
    - Remove macro
    - Improve API
  
  * - ``using ActiveExecutionMemorySpace``
    - Remove type alias
    - Improve API
  
  * - ``Kokkos::Experimental::aMathFunction``
    - Use ``namespace Kokkos``
    - Promote to Kokkos namespace
  
  * - ``Kokkos::Experimental::clamp``
    - Use ``namespace Kokkos``
    - Promote to Kokkos namespace
  
  * - ``Kokkos::Experimental::max;``
    - Use ``namespace Kokkos``
    - Promote to Kokkos namespace
  
  * - ``Kokkos::Experimental::min``
    - Use ``namespace Kokkos``
    - Promote to Kokkos namespace
  
  * - ``Kokkos::Experimental::minmax``
    - Use `namespace Kokkos`
    - Promote to Kokkos namespace
  
  * - ``using Iterate``
    - Remove type alias
    - Improve API
  
  * - ``using MDRangePolicy``
    - Remove type alias
    - Improve API
  
  * - ``using Rank``
    - Remove type alias
    - Improve API
  
  * - Test reduction of a pointer to a 1D array ``parallel_reduce(range, functor, sums_ptr)``
    - Remove test
    - Update testing
  
  * - ``void take_initialization_settings(Kokkos::InitializationSettings const&) {}``
    - Remove test
    - Update testing
  
  * - Test scalar result in host pointer in ``parallel_reduce`` ``(ASSERT_EQ(host_result(j), (ScalarType)correct);``
    - Remove test case
    - Update testing
  
  * - ``Kokkos::parallel_reduce(policy, ReducerWithJoinThatTakesVolatileQualifiedArgs{}, result);``
    - Remove test case
    - Update testing

  * - ``TEST(openmp, partition_master)``
    - Remove test
    - Update testing
