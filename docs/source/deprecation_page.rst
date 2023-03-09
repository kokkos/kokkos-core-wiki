.. include:: mydefs.rst

Deprecation
===========

Kokkos-3.x
----------

.. list-table::
   :widths: 30 70
   :header-rows: 1
   :align: left

   * - Previous
     - New

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

   * - ``!std::is_empty<typename base_t::work_tag>::value`` && ``!std::is_void<typename base_t::work_tag>::value``
     - Removed

   * - ``: secName(sectionName)`` in ``class ProfilingSection``
     - Removed

   * - ``std::string getName() { return secName; }``
     - Removed

   * - ``uint32_t getSectionID() { return secID; }``
     - Removed

   * - ``const std::string secName``
     - Removed

   * - ``using ActiveExecutionMemorySpace``
     - Removed

   * - ``using ActiveExecutionMemorySpace``
     - Removed

   * - ``using is_array_layout``
     - Removed

   * - ``using is_execution_policy``
     - Removed

   * - ``using is_execution_space``
     - Removed

   * - ``using is_memory_space``
     - Removed

   * - ``using is_memory_traits``
     - Removed

   * - ``using host_memory_space``
     - Removed

   * - ``using host_execution_space``
     - Removed

   * - ``using host_mirror_space``
     - Removed

   * - ``namespace Impl``
     - Removed

   * - ``using is_space``
     - Removed

   * - ``using SpaceAccessibility``
     - Removed

   * - ``#define KOKKOS_RESTRICT_EXECUTION_TO_(DATA_SPACE)``
     - Removed

   * - ``parallel_*`` overloads taking the label as trailing argument
     - ``Kokkos::parallel_*("KokkosViewLabel", policy, f);``

   * - ``InitArguments`` struct
     - ``InitializationSettings()`` class object with query-able attributes

   * - ``finalize_all()``
     - ``finalize()``

   * - Warn about ``parallel_reduce`` cases that call ``join()`` with arguments qualified by ``volatile`` keyword
     - Removed ``volatile`` overloads

   * - ``static void partition_master(F const& f, int requested_num_partitions = 0, int requested_partition_size = 0)``
     - Removed

   * - ``std::vector<OpenMP> OpenMP::partition(...) { return std::vector<OpenMP>(1); }``
     - Removed

   * - ``OpenMP OpenMP::create_instance(...) { return OpenMP(); }``
     - Removed

   * - ``static void validate_partition(const int nthreads, int& num_partitions, int& partition_size)``
     - Removed

   * - ``void OpenMP::partition_master(F const& f, int num_partitions, int partition_size)``
     - Removed

   * - ``class MasterLock<OpenMP>``
     - Removed

   * - ``class KOKKOS_ATTRIBUTE_NODISCARD ScopeGuard``
     - Removed

   * - ``create_mirror_view`` taking ``WithOutInitializing`` as first argument
     - ``create_mirror_view(Kokkos::Impl::WithoutInitializing_t wi, Kokkos::View<T, P...> const& v)``

   * - ``#define KOKKOS_THREAD_LOCAL`` macro
     - ``thread_local``

   * - ``vector_length() const``
     - Removed

   * - ``class MasterLock``
     - Removed

   * - ``Kokkos::Impl::is_view``
     - ``Kokkos::is_view``

   * - ``inline int vector_length() const``
     - Removed

   * - Command-line arguments (other than ``--help``) not prefixed with ``kokkos-*``
     - ``--kokkos-num-threads``, ``--kokkos-device-id``, ``--kokkos-num-devices``, ``--kokkos-numa``, ``--kokkos-num-threads``, ``--kokkos-num-threads``

   * - ``void CudaSpace::access_error()``
     - Removed

   * - ``int CudaUVMSpace::number_of_allocations()``
     - Removed

   * - ``inline void cuda_internal_safe_call_deprecated()``
     - ``#define CUDA_SAFE_CALL(call)``

   * - ``partition(...)``, ``partition_master`` for HPX backend
     - Removed

   * - ``static void access_error();``
     - Removed

   * - ``static void access_error(const void* const);``
     - Removed

   * - ``static int number_of_allocations();``
     - Removed

   * - ``using ActiveExecutionMemorySpace``
     - Removed

   * - ``using ActiveExecutionMemorySpace``
     - Removed

   * - ``void Experimental::HIPSpace::access_error()``
     - Removed

   * - ``void Experimental::HIPSpace::access_error(const void* const)``
     - Removed

   * - ``inline void hip_internal_safe_call_deprecated``
     - Removed

   * - ``#define HIP_SAFE_CALL(call)``
     - Removed

   * - ``using ActiveExecutionMemorySpace``
     - Removed

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

   * - ``using Iterate``
     - Removed

   * - ``using MDRangePolicy``
     - Removed

   * - ``using Rank``
     - Removed

   * - Test reduction of a pointer to a 1D array ``parallel_reduce(range, functor, sums_ptr)``
     - Removed

   * - ``void take_initialization_settings(Kokkos::InitializationSettings const&) {}``
     - Removed

   * - Test scalar result in host pointer in ``parallel_reduce`` ``(ASSERT_EQ(host_result(j), (ScalarType)correct);``
     - Removed

   * - ``Kokkos::parallel_reduce(policy, ReducerWithJoinThatTakesVolatileQualifiedArgs{}, result);``
     - Removed

   * - ``TEST(openmp, partition_master)``
     - Removed


Public and Private Headers
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning:: The following headers are now :red:`private` and therefore should not be included directly:

  ``Kokkos_Bitset.hpp``, ``Kokkos_DualView.hpp``, ``Kokkos_DynRankView.hpp``,
  ``Kokkos_ErrorReporter.hpp``, ``Kokkos_Functional.hpp``, ``Kokkos_OffsetView.hpp``,
  ``Kokkos_ScatterView.hpp``, ``Kokkos_StaticCrsGraph.hpp``, ``Kokkos_UnorderedMap.hpp``,
  ``Kokkos_Vector.hpp``, ``Kokkos_UniqueToken.hpp``, ``Kokkos_Threads.hpp``,
  ``Kokkos_Serial.hpp``, ``Kokkos_AnonymousSpace.hpp``, ``Kokkos_Atomics_Desul_Config.hpp``,
  ``Kokkos_Vectorization.hpp``, ``Kokkos_OpenACC.hpp``, ``Kokkos_OpenACCSpace.hpp``,
  ``Kokkos_MasterLock.hpp``, ``Kokkos_View.hpp``, ``Kokkos_ExecPolicy.hpp``,
  ``Kokkos_Future.hpp``, ``Kokkos_GraphNode.hpp``, ``Kokkos_HBWSpace.hpp``,
  ``Kokkos_ScratchSpace.hpp``, ``Kokkos_Crs.hpp``, ``Kokkos_SYCL_Space.hpp``,
  ``Kokkos_SYCL.hpp``, ``Kokkos_Cuda.hpp``, ``Kokkos_CudaSpace.hpp``,
  ``KokkosExp_MDRangePolicy.hpp``, ``Kokkos_Tuners.hpp``, ``Kokkos_HIP_Space.hpp``,
  ``Kokkos_HIP.hpp``, ``Kokkos_Rank.hpp``, ``Kokkos_Atomics_Desul_Volatile_Wrapper.hpp``,
  ``Kokkos_Atomics_Desul_Wrapper.hpp``, ``Kokkos_MinMaxClamp.hpp``, ``Kokkos_Concepts.hpp``,
  ``Kokkos_MemoryPool.hpp``, ``Kokkos_Parallel_Reduce.hpp``, ``Kokkos_TaskScheduler.hpp``,
  ``Kokkos_TaskScheduler_fwd.hpp``, ``Kokkos_hwloc.hpp``, ``Kokkos_PointerOwnership.hpp``,
  ``Kokkos_OpenMPTarget.hpp``, ``Kokkos_OpenMPTargetSpace.hpp``, ``Kokkos_Layout.hpp``,
  ``Kokkos_MemoryTraits.hpp``, ``Kokkos_LogicalSpaces.hpp``, ``Kokkos_Extents.hpp``,
  ``Kokkos_AcquireUniqueTokenImpl.hpp``, ``Kokkos_CopyViews.hpp``, ``Kokkos_HostSpace.hpp``,
  ``Kokkos_HPX.hpp``, ``Kokkos_OpenMP.hpp``, ``Kokkos_Parallel.hpp``, ``Kokkos_WorkGraphPolicy.hpp``

.. important:: The following headers are :large:`public` headers you should use:

  ``Kokkos_Core.hpp``, ``Kokkos_Macros.hpp``, ``Kokkos_Atomic.hpp``,
  ``Kokkos_DetectionIdiom.hpp``, ``Kokkos_MathematicalConstants.hpp``,
  ``Kokkos_MathematicalFunctions.hpp``, ``Kokkos_NumericTraits.hpp``,
  ``Kokkos_Array.hpp``, ``Kokkos_Complex.hpp``, ``Kokkos_Pair.hpp``,
  ``Kokkos_Half.hpp``, ``Kokkos_Timer.hpp``,
  ``Kokkos_StdAlgorithms.hpp``, ``Kokkos_Random.hpp``, ``Kokkos_Sort.hpp``
