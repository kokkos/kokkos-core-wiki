Deprecations
************

Deprecations for Kokkos 4
=========================

up to 43a30195af81b6d1aa5b1efec939af8423857f2c

- Default constructor for BinOp1D, BinOp3D
- Experimental::swap -> kokkos_swap
- vector
- ExecutionSpace::in_parallel
- Cuda::Cuda(cudaStream_t stream) -> Cuda(cudaStream_t stream, Impl::ManageStream manage_stream)
- Cuda::Cuda(cudaStream_t stream, bool manage_stream) -> Cuda(cudaStream_t stream, Impl::ManageStream manage_stream)
- Cuda::device_arch()
- Cuda::detect_device_count()
- Cuda::detect_device_arch()
- CudaUVMSpace::available()
- HIP::HIP(hipStream_t const stream, bool manage_stream) -> HIP(cudaStream_t stream, Impl::ManageStream manage_stream)
- HIP(hipStream_t st) -> HIP(cudaStream_t stream, Impl::ManageStream manage_stream)
- HIP::detect_device_count()
- HPX(instance_mode mode)
- HPX(hpx::execution::experimental::unique_any_sender<> &&sender)
- HPX::is_asynchronous(HPX const & = HPX())
- Complex volatile overloads
- Tasking
- Array<void, KOKKOS_INVALID_INDEX, void>
- Array<T, KOKKOS_INVALID_INDEX, Impl::KokkosArrayContiguous>
- Array<T, KOKKOS_INVALID_INDEX, Impl::KokkosArrayStrided>
- ChunkSize::ChunkSize(int value) -> explicit ChunkSize::ChunkSize(int value)
- RangePolicy::set(ChunkSize chunksize) -> RangePolicy::set_chunk_size(int chunk_size)
- HostSpace::AllocationMechanism, HostSpace::HostSpace(AllocationMechanism)
- is_layouttiled
- layout_iterate_type_selector
- pair volatile overloads
- pair<T1, void>
- ScratchMemeorySpace::align(const IntType& size)
- OpenMP::OpenMP(int pool_size) -> explicit OpenMP::OpenMP(int pool_size)
- OpenMP::is_asynchronous(OpenMP const& = OpenMP())
- Serial::Serial(NewInstance) -> explicit Serial::Serial(NewInstance)
- View::Rank -> View::rank()
- View::subview<MemoryTraits>(...) -> View::subview(...)
- Impl::ALL_t -> ALL_t
- Reduce/scan join() taking volatile-qualified parameters
- InitializationSettings::set_num_devices, InitializationSettings::has_num_devices, InitializationSettings::get_num_devices
- InitializationSettings::set_skip_devices, InitializationSettings::has_skip_devices, InitializationSettings::get_skip_devices
- SIMD math functions in the Kokkos::Experimental namespace -> SIMD math function in the Kokkos namespace


Deprecations for Kokkos-3.x
===========================



Type aliases deprecated in Kokkos-3.7
-------------------------------------
``ActiveExecutionMemorySpace``, ``host_execution_space``, ``host_memory_space``, ``host_mirror_space``, ``is_array_layout``, ``is_execution_policy``, ``is_execution_space``, ``is_memory_space``, ``is_memory_traits``, ``is_space``, ``Iterate``, ``MDRangePolicy``, ``Rank``, ``SpaceAccessibility``


Macros deprecated in Kokkos-3.7
-------------------------------

``KOKKOS_RESTRICT_EXECUTION_TO_(DATA_SPACE)``, ``HIP_SAFE_CALL(call)``


Free-functions deprecated in Kokkos-3.7
---------------------------------------

.. list-table::  
   :widths: 30 70
   :header-rows: 1

   * - Name 
     - Where

   * - .. code-block:: cpp 

          std::vector<OpenMP> OpenMP::partition(...)

     - OpenMP

   * - .. code-block:: cpp

          OpenMP OpenMP::create_instance(...)

     - OpenMP

   * - .. code-block:: cpp

          void OpenMP::partition_master(F const& f,
                                        int num_partitions,
                                        int partition_size)

     - OpenMP (Kokkos_OpenMP_Instance.hpp)

   * - .. code-block:: cpp

          void Experimental::HIPSpace::access_error()

     - ``namespace Kokkos`` (Kokkos_HIP_Space.cpp)

   * - .. code-block:: cpp

          void Experimental::HIPSpace::access_error(const void* const)

     - ``namespace Kokkos`` (Kokkos_HIP_Space.cpp)

   * - ..  code-block:: cpp

           inline void hip_internal_safe_call_deprecated

     - ``namespace Kokkos::Impl`` (Kokkos_HIP_Error.hpp)


Member functions deprecated in Kokkos-3.7
------------------------------------------

.. list-table::  
   :widths: 70 30
   :header-rows: 1

   * - Method name
     - Class

   * - ``static void OpenMP::partition_master()``
     - ``class OpenMP`` (Kokkos_OpenMP.hpp)

   * - ``static void OpenMPInternal::validate_partition()``
     - ``class OpenMPInternal`` (Kokkos_OpenMP_Instance.hpp)

   * - ``std::string ProfilingSection::getName()``
     - ``class ProfilingSection`` (Kokkos_Profiling_ProfileSection.hpp)

   * - ``uint32_t ProfilingSection::getSectionID()``
     - ``class ProfilingSection`` (Kokkos_Profiling_ProfileSection.hpp)

   * - ``int TeamPolicyInternal::vector_length() const``
     - ``class TeamPolicyInternal`` (Kokkos_HIP_Parallel_Team.hpp, Kokkos_SYCL_Parallel_Team.hpp)

   * - ``inline int TeamPolicyInternal::vector_length() const``
     - ``class TeamPolicyInternal`` (Kokkos_OpenMPTarget_Exec.hpp, Kokkos_Cuda_Parallel_Team.hpp)

   * - ``static void CudaSpace::access_error();``
     - ``class CudaSpace`` (Kokkos_CudaSpace.hpp), ``class HIPSpace`` (Kokkos_HIP_Space.hpp)

   * - ``static void CudaSpace::access_error(const void* const);``
     - ``class CudaSpace`` (Kokkos_CudaSpace.hpp), ``class HIPSpace`` (Kokkos_HIP_Space.hpp)

   * - ``static int CudaUVMSpace::number_of_allocations();``
     - ``class CudaUVMSpace`` (Kokkos_CudaSpace.hpp)

   * - ``HPX::partition(...), HPX::partition_master()`` 
     - ``class HPX`` (Kokkos_HPX.hpp)


Classes deprecated in Kokkos-3.7
--------------------------------

.. list-table::  
   :widths: auto
   :header-rows: 1

   * - 

   * - ``class MasterLock<OpenMP>``

   * - ``class KOKKOS_ATTRIBUTE_NODISCARD ScopeGuard``


Namespace updates
----------------------

.. list-table::  
   :widths: 40 60
   :header-rows: 1

   * - Previous
     - You should now use
 
   * - ``Kokkos::Experimental::aMathFunction``
     - ``Kokkos::aMathFunction``

   * - ``Kokkos::Experimental::clamp``
     - ``Kokkos::clamp``

   * - ``Kokkos::Experimental::max;``
     - ``Kokkos::max``

   * - ``Kokkos::Experimental::min``
     - ``Kokkos::min``

   * - ``Kokkos::Experimental::minmax``
     - ``Kokkos::minmax``


Other deprecations
------------------

.. list-table::  
   :widths: auto
   :header-rows: 1

   * - Previous
     - Replaced with

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

   * - ``InitArguments`` struct
     - ``InitializationSettings()`` class object with query-able attributes

   * - ``finalize_all()``
     - ``finalize()``

   * - Warn about ``parallel_reduce`` cases that call ``join()`` with arguments qualified by ``volatile`` keyword
     - Remove ``volatile`` overloads


   * - ``create_mirror_view`` taking ``WithOutInitializing`` as first argument
     - ``create_mirror_view(Kokkos::Impl::WithoutInitializing_t wi, Kokkos::View<T, P...> const& v)``

   * - ``#define KOKKOS_THREAD_LOCAL`` macro
     - ``thread_local``

   * - ``class MasterLock``
     - Remove class

   * - ``Kokkos::Impl::is_view``
     - ``Kokkos::is_view``

   * - ``inline void cuda_internal_safe_call_deprecated()``
     - ``#define CUDA_SAFE_CALL(call)``

   * - ``parallel_*`` overloads taking the label as trailing argument
     - ``Kokkos::parallel_*("KokkosViewLabel", policy, f);``


Public Headers in Kokkos-3.7 
----------------------------

From Kokkos-3.7, the following are *public* headers:

Core
~~~~~~~~~~~~
``Kokkos_Core.hpp``, ``Kokkos_Macros.hpp``, ``Kokkos_Atomic.hpp``, ``Kokkos_DetectionIdiom.hpp``, ``Kokkos_MathematicalConstants.hpp``, ``Kokkos_MathematicalFunctions.hpp``, ``Kokkos_NumericTraits.hpp``, ``Kokkos_Array.hpp``, ``Kokkos_Complex.hpp``, ``Kokkos_Pair.hpp``, ``Kokkos_Half.hpp``, ``Kokkos_Timer.hpp``

Algorithms
~~~~~~~~~~~~~~~~~~
``Kokkos_StdAlgorithms.hpp``, ``Kokkos_Random.hpp``, ``Kokkos_Sort.hpp``

Containers
~~~~~~~~~~~~~~~~~~
``Kokkos_Bit.hpp``, ``Kokkos_DualView.hpp``, ``Kokkos_DynRankView.hpp``, ``Kokkos_ErrorReporter.hpp``, ``Kokkos_Functional.hpp``, ``Kokkos_OffsetView.hpp``, ``Kokkos_ScatterView.hpp``, ``Kokkos_StaticCrsGraph.hpp``, ``Kokkos_UnorderedMap.hpp``, ``Kokkos_Vector.hpp``   
