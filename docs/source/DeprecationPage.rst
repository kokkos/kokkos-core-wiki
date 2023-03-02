Deprecation
-----------

.. raw:: html

   <!--- VERSION 3 DEPRECATION --->

.. _kokkos-3x:

Kokkos-3.x
----------

+----------------------+----------------------+----------------------+
| **Deprecated**       | **Replacement**      | **Reason**           |
+======================+======================+======================+
| ``Kokko              | ``                   | Improve API          |
| s::is_reducer_type`` | Kokkos::is_reducer`` |                      |
+----------------------+----------------------+----------------------+
| Array reductions     | Use ``Kokkos::View`` | Improve API          |
| with raw pointer     | as return argument   |                      |
+----------------------+----------------------+----------------------+
| ``OffsetView``       | ``Kokkos::pair``     | Streamline arguments |
| constructors taking  | (CPU and GPU)        | to ``::pair``        |
| ``index_list_type``  |                      | function             |
+----------------------+----------------------+----------------------+
| Overloads of         | Use                  | Updating overloads   |
| ``Kokkos::sort``     | ``Kokkos::BinSort``  |                      |
| taking a parameter   | if required, or call |                      |
| ``bool alwa          | ``Kokkos::sort``     |                      |
| ys_use_kokkos_sort`` | without bool         |                      |
|                      | parameter            |                      |
+----------------------+----------------------+----------------------+
| Raise deprecation    | Use empty WorkTag    | Improve API          |
| warnings if          | class                |                      |
| non-empty WorkTag    |                      |                      |
| class is used        |                      |                      |
+----------------------+----------------------+----------------------+
| ``: se               | Remove constructor   | Improve API          |
| cName(sectionName)`` |                      |                      |
| in                   |                      |                      |
| ``clas               |                      |                      |
| s ProfilingSection`` |                      |                      |
+----------------------+----------------------+----------------------+
| ``std                | Remove function      | Improve API          |
| ::string getName() { |                      |                      |
|  return secName; }`` |                      |                      |
+----------------------+----------------------+----------------------+
| ``uin                | Remove function      | Improve API          |
| t32_t getSectionID() |                      |                      |
|  { return secID; }`` |                      |                      |
+----------------------+----------------------+----------------------+
| ``const st           | Remove variable      | Improve API          |
| d::string secName;`` |                      |                      |
+----------------------+----------------------+----------------------+
| ``using ActiveEx     | Remove type alias    | Improve API          |
| ecutionMemorySpace`` |                      |                      |
+----------------------+----------------------+----------------------+
| ``using ActiveEx     | Remove type alias    | Improve API          |
| ecutionMemorySpace`` |                      |                      |
+----------------------+----------------------+----------------------+
| ``usi                | Remove type alias    | Improve API          |
| ng is_array_layout`` |                      |                      |
+----------------------+----------------------+----------------------+
| ``using i            | Remove type alias    | Improve API          |
| s_execution_policy`` |                      |                      |
+----------------------+----------------------+----------------------+
| ``using              | Remove type alias    | Improve API          |
| is_execution_space`` |                      |                      |
+----------------------+----------------------+----------------------+
| ``usi                | Remove type alias    | Improve API          |
| ng is_memory_space`` |                      |                      |
+----------------------+----------------------+----------------------+
| ``usin               | Remove type alias    | Improve API          |
| g is_memory_traits`` |                      |                      |
+----------------------+----------------------+----------------------+
| ``using              | Remove type alias    | Improve API          |
|  host_memory_space`` |                      |                      |
+----------------------+----------------------+----------------------+
| ``using ho           | Remove type alias    | Improve API          |
| st_execution_space`` |                      |                      |
+----------------------+----------------------+----------------------+
| ``using              | Remove type alias    | Improve API          |
|  host_mirror_space`` |                      |                      |
+----------------------+----------------------+----------------------+
| ``namespace Impl``   | Remove               | Improve API          |
|                      | ``namespace Impl``   |                      |
+----------------------+----------------------+----------------------+
| ``using is_space``   | Remove type alias    | Improve API          |
+----------------------+----------------------+----------------------+
| ``using              | Remove type alias    | Improve API          |
| SpaceAccessibility`` |                      |                      |
+----------------------+----------------------+----------------------+
| ``#define KOK        | Remove macro         | Improve API          |
| KOS_RESTRICT_EXECUTI |                      |                      |
| ON_TO_(DATA_SPACE)`` |                      |                      |
+----------------------+----------------------+----------------------+
| ``parallel_*``       | ``Kokkos::par        | Consistent ordering  |
| overloads taking the | allel_*("KokkosViewL | of parameters        |
| label as trailing    | abel", policy, f);`` |                      |
| argument             |                      |                      |
+----------------------+----------------------+----------------------+
| ``InitArguments``    | ``Initia             | Verifiable           |
| struct               | lizationSettings()`` | initialization       |
|                      | class object with    |                      |
|                      | query-able           |                      |
|                      | attributes           |                      |
+----------------------+----------------------+----------------------+
| ``finalize_all()``   | ``finalize()``       | Improve API          |
+----------------------+----------------------+----------------------+
| Warn about           | Remove ``volatile``  | Streamline API       |
| ``parallel_reduce``  | overloads            |                      |
| cases that call      |                      |                      |
| ``join()`` with      |                      |                      |
| arguments qualified  |                      |                      |
| by ``volatile``      |                      |                      |
| keyword              |                      |                      |
+----------------------+----------------------+----------------------+
| ``static voi         | Remove function      | Improve API          |
| d partition_master(F |                      |                      |
|  const& f, int reque |                      |                      |
| sted_num_partitions  |                      |                      |
| = 0, int requested_p |                      |                      |
| artition_size = 0)`` |                      |                      |
+----------------------+----------------------+----------------------+
| `                    | Remove function      | Improve API          |
| `std::vector<OpenMP> |                      |                      |
|  OpenMP::partition(. |                      |                      |
| ..) { return std::ve |                      |                      |
| ctor<OpenMP>(1); }`` |                      |                      |
+----------------------+----------------------+----------------------+
| ``OpenMP OpenMP::cre | Remove function      | Improve API          |
| ate_instance(...) {  |                      |                      |
| return OpenMP(); }`` |                      |                      |
+----------------------+----------------------+----------------------+
| ``static void va     | Remove function      | Improve API          |
| lidate_partition(con |                      |                      |
| st int nthreads, int |                      |                      |
| & num_partitions, in |                      |                      |
| t& partition_size)`` |                      |                      |
+----------------------+----------------------+----------------------+
| ``voi                | Remove function      | Improve API          |
| d OpenMP::partition_ |                      |                      |
| master(F const& f, i |                      |                      |
| nt num_partitions, i |                      |                      |
| nt partition_size)`` |                      |                      |
+----------------------+----------------------+----------------------+
| ``class              | Remove class         | Improve API          |
| MasterLock<OpenMP>`` |                      |                      |
+----------------------+----------------------+----------------------+
| ``class              | Remove class         | Improve API          |
|  KOKKOS_ATTRIBUTE_NO |                      |                      |
| DISCARD ScopeGuard`` |                      |                      |
+----------------------+----------------------+----------------------+
| ``                   | ``create_mirro       | Improve API          |
| create_mirror_view`` | r_view(Kokkos::Impl: |                      |
| taking               | :WithoutInitializing |                      |
| ``W                  | _t wi, Kokkos::View< |                      |
| ithOutInitializing`` | T, P...> const& v)`` |                      |
| as first argument    |                      |                      |
+----------------------+----------------------+----------------------+
| ``!                  | Remove condition     | Improve API          |
| std::is_empty<typena |                      |                      |
| me base_t::work_tag> |                      |                      |
| ::value && !std::is_ |                      |                      |
| void<typename base_t |                      |                      |
| ::work_tag>::value`` |                      |                      |
+----------------------+----------------------+----------------------+
| ``partition(...)``,  | Remove function      | Improve API          |
| ``partition_master`` |                      |                      |
| for HPX backend      |                      |                      |
+----------------------+----------------------+----------------------+
| ``constexpr``        | Remove specifier     | Improve API          |
+----------------------+----------------------+----------------------+
| ``#define K          | ``thread_local``     | Improve API          |
| OKKOS_THREAD_LOCAL`` |                      |                      |
| macro                |                      |                      |
+----------------------+----------------------+----------------------+
| ``vec                | Remove function      | Improve API          |
| tor_length() const`` |                      |                      |
+----------------------+----------------------+----------------------+
| ``class MasterLock`` | Remove class         | Improve API          |
+----------------------+----------------------+----------------------+
| ``Kok                | ``Kokkos::is_view``  | Improve API          |
| kos::Impl::is_view`` |                      |                      |
+----------------------+----------------------+----------------------+
| ``inline int vec     | Remove function      | Improve API          |
| tor_length() const`` |                      |                      |
+----------------------+----------------------+----------------------+

Kokkos Public Headers
=====================

Kokkos Core
-----------

\| Including private headers is deprecated \| ``Kokkos_Core.hpp``,
``Kokkos_Macros.hpp``, ``Kokkos_Atomic.hpp``,
``Kokkos_DetectionIdiom.hpp``, ``Kokkos_MathematicalConstants.hpp``,
``Kokkos_MathematicalFunctions.hpp``, ``Kokkos_NumericTraits.hpp``,
``Kokkos_Array.hpp``, ``Kokkos_Complex.hpp``, ``Kokkos_Pair.hpp``,
``Kokkos_Half.hpp``, ``Kokkos_Timer.hpp`` \| Improve API

Kokkos Algorithms
-----------------

\| Including private headers is deprecated \|
``Kokkos_StdAlgorithms.hpp``, ``Kokkos_Random.hpp``, ``Kokkos_Sort.hpp``
\| Improve API

Kokkos Containers
-----------------

\| Including private headers is deprecated \| ``Kokkos_Bitset.hpp``, \|
\| ``Kokkos_DualView.hpp``, ``Kokkos_DynRankView.hpp``,
``Kokkos_ErrorReporter.hpp``, ``Kokkos_Functional.hpp``,
``Kokkos_OffsetView.hpp``, ``Kokkos_ScatterView.hpp``,
``Kokkos_StaticCrsGraph.hpp``, ``Kokkos_UnorderedMap.hpp``,
``Kokkos_Vector.hpp``

Kokkos Private Headers
======================

\| ``Kokkos_UniqueToken.hpp``, ``Kokkos_Threads.hpp``,
``Kokkos_Serial.hpp``, ``Kokkos_AnonymousSpace.hpp``,
``Kokkos_Atomics_Desul_Config.hpp``, ``Kokkos_Vectorization.hpp``,
``Kokkos_OpenACC.hpp``, ``Kokkos_OpenACCSpace.hpp``,
``Kokkos_MasterLock.hpp``, ``Kokkos_View.hpp``,
``Kokkos_ExecPolicy.hpp``, ``Kokkos_Future.hpp``,
``Kokkos_GraphNode.hpp``, ``Kokkos_HBWSpace.hpp``,
``Kokkos_ScratchSpace.hpp``, ``Kokkos_Crs.hpp``,
``Kokkos_SYCL_Space.hpp``, ``Kokkos_SYCL.hpp``, ``Kokkos_Cuda.hpp``,
``Kokkos_CudaSpace.hpp``, ``KokkosExp_MDRangePolicy.hpp``,
``Kokkos_Tuners.hpp``, ``Kokkos_HIP_Space.hpp``, ``Kokkos_HIP.hpp``,
``Kokkos_Rank.hpp``, ``Kokkos_Atomics_Desul_Volatile_Wrapper.hpp``,
``Kokkos_Atomics_Desul_Wrapper.hpp``, ``Kokkos_MinMaxClamp.hpp``,
``Kokkos_Concepts.hpp``, ``Kokkos_MemoryPool.hpp``,
``Kokkos_Parallel_Reduce.hpp``, ``Kokkos_TaskScheduler.hpp``,
``Kokkos_TaskScheduler_fwd.hpp``, ``Kokkos_hwloc.hpp``,
``Kokkos_PointerOwnership.hpp``, ``Kokkos_OpenMPTarget.hpp``,
``Kokkos_OpenMPTargetSpace.hpp``, ``Kokkos_Layout.hpp``,
``Kokkos_MemoryTraits.hpp``, ``Kokkos_LogicalSpaces.hpp``,
``Kokkos_Extents.hpp``, ``Kokkos_AcquireUniqueTokenImpl.hpp``,
``Kokkos_CopyViews.hpp``, ``Kokkos_HostSpace.hpp``, ``Kokkos_HPX.hpp``,
``Kokkos_OpenMP.hpp``, ``Kokkos_Parallel.hpp``,
``Kokkos_WorkGraphPolicy.hpp`` \| ``Kokkos_Core.hpp`` \| Improve API

Command-Line Arguments Updates
==============================

\| Command-line arguments (other than ``--help``) not prefixed with
``kokkos-*`` \| ``--kokkos-num-threads``, ``--kokkos-device-id``,
``--kokkos-num-devices``, ``--kokkos-numa``, ``--kokkos-num-threads``,
``--kokkos-num-threads`` \| Improve API

Backends
========

CUDA Deprecation
----------------

\| ``void CudaSpace::access_error()`` \| Remove function \| Improve API
\| ``int CudaUVMSpace::number_of_allocations()`` \| Remove function \|
Improve API \| ``inline void cuda_internal_safe_call_deprecated()`` \|
``#define CUDA_SAFE_CALL(call)`` \| Improve API \|
``static void access_error();`` \| Remove function \| Improve API \|
``static void access_error(const void* const);`` \| Remove function \|
``static int number_of_allocations();`` \| Remove function \| Improve
API \| ``using ActiveExecutionMemorySpace`` \| Remove type alias \|
Improve API

HIP Deprecation
---------------

\| ``using ActiveExecutionMemorySpace`` \| Remove type alias \| Improve
API \| ``void Experimental::HIPSpace::access_error()`` \| Remove
function \| Improve API \|
``void Experimental::HIPSpace::access_error(const void* const)`` \|
Remove function \| Improve API \|
``inline void hip_internal_safe_call_deprecated  |  Remove function  |  Improve API |``\ #define
HIP_SAFE_CALL(call)\` \| Remove macro \| Improve API

SYCL Deprecation
----------------

\| ``using ActiveExecutionMemorySpace`` \| Remove type alias \| Improve
API

Namespace Changes
=================

Promotion to Kokkos Namespace
-----------------------------

\| ``Kokkos::Experimental::aMathFunction`` \| Use ``namespace Kokkos``
\| Promote to Kokkos namespace \| ``Kokkos::Experimental::clamp`` \| Use
``namespace Kokkos`` \| Promote to Kokkos namespace \|
``Kokkos::Experimental::max;`` \| Use ``namespace Kokkos`` \| Promote to
Kokkos namespace \| ``Kokkos::Experimental::min;`` \| Use
``namespace Kokkos`` \| Promote to Kokkos namespace \|
``Kokkos::Experimental::minmax;`` \| Use ``namespace Kokkos`` \| Promote
to Kokkos namespace \| ``using Iterate`` \| Remove type alias \| Improve
API \| ``using MDRangePolicy`` \| Remove type alias \| Improve API \|
``using Rank`` \| Remove type alias \| Improve API

Testing
=======

Unit Test
---------

\| Test reduction of a pointer to a 1D array
``parallel_reduce(range, functor, sums_ptr)`` \| Remove test \| Update
testing \|
``void take_initialization_settings(Kokkos::InitializationSettings const&) {}``
\| Remove test \| Update testing \| Test scalar result in host pointer
in ``parallel_reduce``
``(ASSERT_EQ(host_result(j), (ScalarType)correct);`` \| Remove test case
\| Update testing \| Kokkos::parallel_reduce(policy,
ReducerWithJoinThatTakesVolatileQualifiedArgs{}, result); \| Remove test
case \| Update testing \| ``TEST(openmp, partition_master)`` \| Remove
test \| Update testing
