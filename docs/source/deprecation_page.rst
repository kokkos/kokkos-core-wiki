Deprecation for Kokkos-3.x
-----------

.. IMPORTANT::
   Kokkos-4.0 requires C++17


A number of Kokkos headers were made private in Kokkos-3.7.

**Kokkos Core public headers:**  ``Kokkos_core.cpp``, ``Kokkos_Macros.hpp``, ``Kokkos_Atomic.hpp``, ``Kokkos_DetectionIdiom``, ``Kokkos_MathematicalConstants.hpp``, ``Kokkos_MathematicalFunctions.hpp``, ``Kokkos_NumericTraits.hpp``, ``Kokkos_Array.hpp``, ``Kokkos_Complex.hpp``, ``Kokkos_Pair.hpp``, ``Kokkos_Half.hpp``, ``Kokkos_Timer.hpp``;

**Kokkos Algorithms public headers:**  ``Kokkos_StdAlgorithms.hpp``, ``Kokkos_Random.hpp``, ``Kokkos_Sort.hpp``;

**Kokkos Containers public headers:**  ``Kokkos_Bit.hpp``, ``Kokkos_DualView.hpp``, ``Kokkos_DynRankView.hpp``, ``Kokkos_ErrorReporter.hpp``, ``Kokkos_Functional.hpp``, ``Kokkos_OffsetView.hpp``, ``Kokkos_ScatterView.hpp``, ``Kokkos_StaticCrsGraph.hpp``, ``Kokkos_UnorderedMap.hpp``, ``Kokkos_Vector.hpp``   


.. list-table::  
   :widths: auto 
   :header-rows: 1

   * - Previous
     - Replaced with
     - Reason

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
