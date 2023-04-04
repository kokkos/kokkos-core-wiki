Deprecation for Kokkos-3.x
==========================

Public Headers in Kokkos-3.7 
----------------------------

Starting from Kokkos-3.7, the following are *public* headers:

Core Library 
~~~~~~~~~~~~
``Kokkos_core.cpp``, ``Kokkos_Macros.hpp``, ``Kokkos_Atomic.hpp``, ``Kokkos_DetectionIdiom``, ``Kokkos_MathematicalConstants.hpp``, ``Kokkos_MathematicalFunctions.hpp``, ``Kokkos_NumericTraits.hpp``, ``Kokkos_Array.hpp``, ``Kokkos_Complex.hpp``, ``Kokkos_Pair.hpp``, ``Kokkos_Half.hpp``, ``Kokkos_Timer.hpp``

Algorithms Library
~~~~~~~~~~~~~~~~~~
``Kokkos_StdAlgorithms.hpp``, ``Kokkos_Random.hpp``, ``Kokkos_Sort.hpp``

Containers Library
~~~~~~~~~~~~~~~~~~
``Kokkos_Bit.hpp``, ``Kokkos_DualView.hpp``, ``Kokkos_DynRankView.hpp``, ``Kokkos_ErrorReporter.hpp``, ``Kokkos_Functional.hpp``, ``Kokkos_OffsetView.hpp``, ``Kokkos_ScatterView.hpp``, ``Kokkos_StaticCrsGraph.hpp``, ``Kokkos_UnorderedMap.hpp``, ``Kokkos_Vector.hpp``   


Type aliases deprecated in Kokkos-3.7
-------------------------------------
``ActiveExecutionMemorySpace``, ``ActiveExecutionMemorySpace``, ``is_array_layout``, ``is_execution_policy``, ``is_execution_space``, ``is_memory_space``, ``is_memory_traits``, ``host_memory_space``, ``host_execution_space``, ``host_mirror_space``, ``is_space``, ``SpaceAccessibility``, ``ActiveExecutionMemorySpace``, ``ActiveExecutionMemorySpace``, ``ActiveExecutionMemorySpace``, ``Iterate``, ``MDRangePolicy``, ``Rank``


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

          static void partition_master(F const& f, 
                                      int requested_num_partitions = 0, 
                                      int requested_partition_size = 0)
     - TBD

   * - .. code-block:: cpp 

          std::vector<OpenMP> OpenMP::partition(...) { 
            return std::vector<OpenMP>(1); 
          }

     - OpenMP

   * - .. code-block:: cpp 

          OpenMP OpenMP::create_instance(...) { 
            return OpenMP(); 
          }

     - OpenMP

   * - .. code-block:: cpp 

          static void validate_partition(const int nthreads, 
                                         int& num_partitions, 
                                         int& partition_size)

     - TBD

   * - .. code-block:: cpp

          void OpenMP::partition_master(F const& f, 
                                        int num_partitions, 
                                        int partition_size)

     - OpenMP

   * - .. code-block:: cpp 

          partition(...), partition_master() 

     - HPX backend


Member functions deprecated in Kokkos-3.7
------------------------------------------

.. list-table::  
   :widths: 70 30
   :header-rows: 1

   * - Method name
     - Class

   * - ``std::string getName() { return secName; }``
     - TBD

   * - ``uint32_t getSectionID() { return secID; }``
     - TBD

   * - ``vector_length() const``
     - TBD

   * - ``inline int vector_length() const``
     - TBD

   * - ``void CudaSpace::access_error()``
     - TBD

   * - ``int CudaUVMSpace::number_of_allocations()``
     - TBD

   * - ``static void access_error();``
     - TBD

   * - ``static void access_error(const void* const);``
     - TBD

   * - ``static int number_of_allocations();``
     - TBD

   * - ``void Experimental::HIPSpace::access_error()``
     - TBD

   * - ``void Experimental::HIPSpace::access_error(const void* const)``
     - TBD

   * - ``inline void hip_internal_safe_call_deprecated``
     - TBD

   * - ``: secName(sectionName)`` in ``class ProfilingSection``
     - TBD


Classes deprecated in Kokkos-3.7
--------------------------------

.. list-table::  
   :widths: auto
   :header-rows: 1

   * - 

   * - ``class MasterLock<OpenMP>``

   * - ``class KOKKOS_ATTRIBUTE_NODISCARD ScopeGuard``


Namespace replacements
----------------------

.. list-table::  
   :widths: 40 60
   :header-rows: 1

   * - Previous
     - You should now use
 
   * - ``Kokkos::Experimental::aMathFunction``
     - ``namespace Kokkos``

   * - ``Kokkos::Experimental::clamp``
     - ``namespace Kokkos``

   * - ``Kokkos::Experimental::max;``
     - ``namespace Kokkos``

   * - ``Kokkos::Experimental::min``
     - ``namespace Kokkos``

   * - ``Kokkos::Experimental::minmax``
     - ``namespace Kokkos``


Tests removed
-------------

.. list-table::  
   :widths: auto
   :header-rows: 1

   * - 

   * - Test reduction of a pointer to a 1D array ``parallel_reduce(range, functor, sums_ptr)``

   * - ``void take_initialization_settings(Kokkos::InitializationSettings const&) {}``

   * - Test scalar result in host pointer in ``parallel_reduce`` ``(ASSERT_EQ(host_result(j), (ScalarType)correct);``

   * - ``Kokkos::parallel_reduce(policy, ReducerWithJoinThatTakesVolatileQualifiedArgs{}, result);``

   * - ``TEST(openmp, partition_master)``


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

   * - ``!std::is_empty<typename base_t::work_tag>::value && !std::is_void<typename base_t::work_tag>::value``
     - Remove condition

   * - ``const std::string secName``
     - Remove variable

   * - ``InitArguments`` struct
     - ``InitializationSettings()`` class object with query-able attributes

   * - ``finalize_all()``
     - ``finalize()``

   * - Warn about ``parallel_reduce`` cases that call ``join()`` with arguments qualified by ``volatile`` keyword
     - Remove ``volatile`` overloads


   * - ``create_mirror_view`` taking ``WithOutInitializing`` as first argument
     - ``create_mirror_view(Kokkos::Impl::WithoutInitializing_t wi, Kokkos::View<T, P...> const& v)``

   * - ``constexpr``
     - Remove specifier

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













