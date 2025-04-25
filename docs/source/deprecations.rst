Deprecations
************

Deprecated in Kokkos 4.x
===========================


Deprecated in Kokkos 4.6
---------------------------

* ``StaticCrsGraph`` moved to Kokkos Kernels
   * replacement: none
   * affinity

* ``native_simd`` type and ``simd_abi`` interface
   * replacement: none
   * alignment with the c++ standard

* Makefile support
   * replacement: CMake
   * transition to a modern build system

* Direct access to ``d_view`` and ``h_view`` members in ``DualView``
   * replacement: ``view_host()`` and ``view_device()``
   * prevent assignment of a view

Deprecated in Kokkos 4.5
---------------------------

* Tasking interface: ``BasicFuture``, ``TaskSingle``, ``TaskPriority``, ``task_spawn``, ``host_spawn``, ``respawn``, ``when_all``, ``wait``
   * replacement: none
   * unused, limited implementation

* ``HPX::HPX::is_asynchronous(HPX const & = HPX())``
   * replacement: none
   * unused, conformity of execution spaces

* ``OpenMP::is_asynchronous(OpenMP const& = OpenMP())``
   * replacement: none
   * unused, conformity of execution spaces

* ``atomic_query_version``
   * replacement: none
   * no known use case

* ``atomic_assign``
   * replacement: ``atomic_store``
   * duplicated functionality

* ``atomic_increment``
   * replacement: ``atoimc_inc``
   * duplicated functionality

* ``atomic_decremnent``
   * replacement: ``atomic_dec``
   * duplicated functionality

* ``atomic_compare_exchange_strong``
   * replacement: ``atomic_compare_exchange``
   * duplicated functionality

Deprecated in Kokkos 4.4
---------------------------

* ``is_layouttiled``
   * replacement: none
   * unused

* ``layout_iterate_type_selector``
   * replacement: none
   * only useful internally

* ``Array<T, N, Proxy>``
   * replacement: none
   * alignment with std::array

* ``HPX::HPX(instance_mode mode)``
   * replacement: ``explicit HPX(instance_mode mode)``
   * execution space instance constructors should be ``explicit``

* ``HPX::HPX(hpx::execution::experimental::unique_any_sender<> &&sender)``
   * replacement: ``explicit HPX::HPX(hpx::execution::experimental::unique_any_sender<> &&sender)``
   * execution space instance constructors should be ``explicit``

* ``OpenMP::OpenMP(int pool_size)``
   * replacement: ``explicit OpenMP::OpenMP(int pool_size)``
   * execution space instance constructors should be ``explicit``

* ``Serial::Serial(NewInstance)``
   * replacement: ``explicit Serial::Serial(NewInstance)``
   * execution space instance constructors should be ``explicit``

* ``ChunkSize::ChunkSize(int value)``
   * replacement: ``explicit ChunkSize::ChunkSize(int value)``
   * ``ChunkSize`` should be constructed explicitly

* ``pair<T, void>``
   * replacement: none
   * the specialization is not documented, does not follow the standard library, it is not tested and has no known usage


Deprecated in Kokkos 4.3
---------------------------

* ``Experimental::swap``
   * replacement: ``kokkos_swap``
   * avoiding ambiguities due to ADL

* ``ExecutionSpace::in_parallel``
   * replacement: ``KOKKOS_IF_ON_HOST``/``KOKKOS_IF_ON_DEVICE`` partly provide similar behavior
   * inconsistent implementation, limited use

* ``Cuda::device_arch()``
   * replacement: none
   * uniformity between execution spaces

* ``Cuda::detect_device_count()``
   * replacement: num_devices()
   * uniformity between execution spaces

* ``Cuda::detect_device_arch()``
   * replacement: none
   * uniformity between execution spaces

* ``HIP::HIP::detect_device_count()``
   * replacement: ``num_devices()``
   * uniformity between execution spaces

* ``RangePolicy::set(ChunkSize chunksize)``
   * replacement: ``RangePolicy::set_chunk_size(int chunk_size)``
   * ``ChunkSize`` was the only extra parameter usable with ``RangePolicy::set()`` 

* ``InitializationSettings::set_num_devices``, ``InitializationSettings::has_num_devices``, ``InitializationSettings::get_num_devices``
   * replacement: ``num_devices``
   * changes in `InitializationSettings` made these superfluous

* ``InitializationSettings::set_skip_devices``, ``InitializationSettings::has_skip_devices``, ``InitializationSettings::get_skip_devices``
   * replacement: ``KOKKOS_VISIBLE_DEVICES``
   * changes in `InitializationSettings` made these superfluous


Deprecated in Kokkos 4.2
---------------------------

* ``Cuda::Cuda(cudaStream_t stream, bool manage_stream)``
   * replacement: ``Cuda::Cuda(cudaStream_t stream)``
   * constructing a Cuda execution space instance should always use an externally managed ``cudaStream`` object
   
* ``HIP::HIP(hipStream_t stream, bool manage_stream)``
    * replacement ``HIP::HIP(hipStream_t stream)``
    * constructing a HIP execution space instance should always use an externally managed ``hipStream`` object
    
* ``vector``
    * replacement: none
    * non-standard behavior, doesn't work well with Kokkos concepts 

* ``HostSpace::HostSpace(AllocationMechanism)``
    * replacement: ``HostSpace::HostSpace()``
    * ``AllocationMechanism`` is unused, ``operator new`` with alignment is used unconditionally

* SIMD math functions in the ``Kokkos::Experimental`` namespace
    * replacement: SIMD math function in the ``Kokkos`` namespace
    * issues with ADL, consistency with other math function overloads 


Deprecated in Kokkos 4.1
---------------------------

* Default constructor for ``BinSort``, ``BinOp1D``, and ``BinOp3D``
   * replacement: none
   * the default constructors created invalid, unusable objects

* ``View::Rank``
   * replacement: ``View::rank()``
   * undocumented, redundant due to existence of ``View::rank()``

* ``View::subview<MemoryTraits>(...)``
   * replacement: ``View::subview(...)``
   * not useful, unused


Deprecated in Kokkos 4.0
---------------------------

* ``CudaUVMSpace::available()``
   * replacement: ``SharedSpace``
   * not portable, would always return ``true``

* ``Complex`` ``volatile`` overloads
   * replacement: none
   * no need for using ``volatile`` overloads

* ``pair`` ``volatile`` overloads
   * replacement: none
   * no need for using ``volatile`` overloads

* ``ScratchMemorySpace::align(const IntType& size)``
   * replacement: none
   * unused, not useful


Deprecated in Kokkos-3.x
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
