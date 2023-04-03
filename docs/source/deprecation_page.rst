Deprecation for Kokkos-3.x
==========================



Kokkos-3.7 Public Headers
-------------------------
   |**Kokkos Core:**  ``Kokkos_core.cpp``, ``Kokkos_Macros.hpp``, ``Kokkos_Atomic.hpp``, ``Kokkos_DetectionIdiom``, ``Kokkos_MathematicalConstants.hpp``, ``Kokkos_MathematicalFunctions.hpp``, ``Kokkos_NumericTraits.hpp``, ``Kokkos_Array.hpp``, ``Kokkos_Complex.hpp``, ``Kokkos_Pair.hpp``, ``Kokkos_Half.hpp``, ``Kokkos_Timer.hpp``;
   |**Kokkos Algorithms:**  ``Kokkos_StdAlgorithms.hpp``, ``Kokkos_Random.hpp``, ``Kokkos_Sort.hpp``;
   |**Kokkos Containers:**  ``Kokkos_Bit.hpp``, ``Kokkos_DualView.hpp``, ``Kokkos_DynRankView.hpp``, ``Kokkos_ErrorReporter.hpp``, ``Kokkos_Functional.hpp``, ``Kokkos_OffsetView.hpp``, ``Kokkos_ScatterView.hpp``, ``Kokkos_StaticCrsGraph.hpp``, ``Kokkos_UnorderedMap.hpp``, ``Kokkos_Vector.hpp``   


Type aliases deprecated in Kokkos-3.7
-------------------------------------
``using ActiveExecutionMemorySpace``, ``using ActiveExecutionMemorySpace``, ``using is_array_layout``, ``using is_execution_policy``, ``using is_execution_space``, ``using is_memory_space``, ``using is_memory_traits``, ``using host_memory_space``, ``using host_execution_space``, ``using host_mirror_space``, ``using is_space``, ``using SpaceAccessibility``, ``using ActiveExecutionMemorySpace``, ``using ActiveExecutionMemorySpace``, ``using ActiveExecutionMemorySpace``, ``using Iterate``, ``using MDRangePolicy``, ``using Rank``


.. list-table::  
   :widths: auto 
   :header-rows: 1

   * - Previous
     - Replaced with

   * - ``namespace Impl``
     - Remove ``namespace Impl``
   
   * - ``Kokkos::Experimental::aMathFunction``
     - Use ``namespace Kokkos``

   * - ``Kokkos::Experimental::clamp``
     - Use ``namespace Kokkos``

   * - ``Kokkos::Experimental::max;``
     - Use ``namespace Kokkos``

   * - ``Kokkos::Experimental::min``
     - Use ``namespace Kokkos``

   * - ``Kokkos::Experimental::minmax``
     - Use `namespace Kokkos`
     - Remove ``namespace Impl`

   * - ``Kokkos::is_reducer_type``
     - ``Kokkos::is_reducer``

   * - Array reductions with raw pointer
     - Use ``Kokkos::View`` as return argument

   * - ``OffsetView`` constructors taking ``index_list_type``
     - ``Kokkos::pair`` (CPU and GPU)

   * - Overloads of ``Kokkos::sort`` taking a parameter ``bool always_use_kokkos_sort``
     - Use ``Kokkos::BinSort`` if required, or call ``Kokkos::sort`` without bool parameter

   * - Raise deprecation warnings if non-empty WorkTag class is used
     - Use empty WorkTag class

   * - ``!std::is_empty<typename base_t::work_tag>::value && !std::is_void<typename base_t::work_tag>::value``
     - Remove condition

   * - ``: secName(sectionName)`` in ``class ProfilingSection``
     - Remove constructor

   * - ``std::string getName() { return secName; }``
     - Remove function

   * - ``uint32_t getSectionID() { return secID; }``
     - Remove function

   * - ``const std::string secName``
     - Remove variable

   * - ``#define KOKKOS_RESTRICT_EXECUTION_TO_(DATA_SPACE)``
     - Remove macro

   * - ``parallel_*`` overloads taking the label as trailing argument
     - ``Kokkos::parallel_*("KokkosViewLabel", policy, f);``

   * - ``InitArguments`` struct
     - ``InitializationSettings()`` class object with query-able attributes

   * - ``finalize_all()``
     - ``finalize()``

   * - Warn about ``parallel_reduce`` cases that call ``join()`` with arguments qualified by ``volatile`` keyword
     - Remove ``volatile`` overloads

   * - ``static void partition_master(F const& f, int requested_num_partitions = 0, int requested_partition_size = 0)``
     - Remove function

   * - ``std::vector<OpenMP> OpenMP::partition(...) { return std::vector<OpenMP>(1); }``
     - Remove function

   * - ``OpenMP OpenMP::create_instance(...) { return OpenMP(); }``
     - Remove function

   * - ``static void validate_partition(const int nthreads, int& num_partitions, int& partition_size)``
     - Remove function

   * - ``void OpenMP::partition_master(F const& f, int num_partitions, int partition_size)``
     - Remove function

   * - ``class MasterLock<OpenMP>``
     - Remove class

   * - ``class KOKKOS_ATTRIBUTE_NODISCARD ScopeGuard``
     - Remove class

   * - ``create_mirror_view`` taking ``WithOutInitializing`` as first argument
     - ``create_mirror_view(Kokkos::Impl::WithoutInitializing_t wi, Kokkos::View<T, P...> const& v)``

   * - ``constexpr``
     - Remove specifier

   * - ``#define KOKKOS_THREAD_LOCAL`` macro
     - ``thread_local``

   * - ``vector_length() const``
     - Remove function

   * - ``class MasterLock``
     - Remove class

   * - ``Kokkos::Impl::is_view``
     - ``Kokkos::is_view``

   * - ``inline int vector_length() const``
     - Remove function

   * - ``void CudaSpace::access_error()``
     - Remove function

   * - ``int CudaUVMSpace::number_of_allocations()``
     - Remove function

   * - ``inline void cuda_internal_safe_call_deprecated()``
     - ``#define CUDA_SAFE_CALL(call)``

   * - ``partition(...)``, ``partition_master`` for HPX backend
     - Remove function 

   * - ``static void access_error();``
     - Remove function

   * - ``static void access_error(const void* const);``
     - Remove function

   * - ``static int number_of_allocations();``
     - Remove function

   * - ``void Experimental::HIPSpace::access_error()``
     - Remove function

   * - ``void Experimental::HIPSpace::access_error(const void* const)``
     - Remove function

   * - ``inline void hip_internal_safe_call_deprecated``
     - Remove function

   * - ``#define HIP_SAFE_CALL(call)``
     - Remove macro

   * - Test reduction of a pointer to a 1D array ``parallel_reduce(range, functor, sums_ptr)``
     - Remove test

   * - ``void take_initialization_settings(Kokkos::InitializationSettings const&) {}``
     - Remove test

   * - Test scalar result in host pointer in ``parallel_reduce`` ``(ASSERT_EQ(host_result(j), (ScalarType)correct);``
     - Remove test case

   * - ``Kokkos::parallel_reduce(policy, ReducerWithJoinThatTakesVolatileQualifiedArgs{}, result);``
     - Remove test case

   * - ``TEST(openmp, partition_master)``
     - Remove test
