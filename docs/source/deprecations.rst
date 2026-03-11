Deprecations
************

Deprecated in Kokkos 5.x
===========================

Deprecated in Kokkos 5.0
---------------------------

* ``KOKKOS_ATTRIBUTE_NODISCARD``
   * replacement: none
   * Not intended for non-internal use.

* ``{Owning,Observing}RawPtr``
   * replacement: none
   * Not intended for non-internal use.

* Nested OpenMP parallel usage without nested OpenMP enabled
   * replacement: none
   * Avoids using a buggy code path when compiled without deprecated code.

* ``OpenMP`` instance creation inside of OpenMP parallel regions
   * replacement: none
   * Avoids ``partition_space`` aborting when the number of threads for a partition is 0.

* ``Random_XorShift{64,1024}_Pool::init``
   * replacmement: none
   * Not intended for non-internal use.

* ``[const_]where_expression``
   * replacement: none
   * Does not align with std::simd interface.

* ``View::HostMirror``
   * replacement: ``View::host_mirror_type``
   * naming style consistency

* ``{DynRankView,OffsetView,View}::scalar_array_type``
   * replacement: ``{DynRankView,OffsetView,View}::data_type``
   * Only relevant for certain external partial specializations of ``{DynRankView, OffsetView, View}`` with the pre Kokkos 5 ``View`` implementation. Equal to ``data_type`` in most cases.

* ``{DynRankView,OffsetView,View}::const_scalar_array_type``
   * replacement: ``{DynRankView,OffsetView,View}::const_data_type``
   * Only relevant for certain external partial specializations of ``{DynRankView, OffsetView, View}`` with the pre Kokkos 5 ``View`` implementation. Equal to ``const_data_type`` in most cases.

* ``{DynRankView,OffsetView,View}::non_const_scalar_array_type``
   * replacement: ``{DynRankView,OffsetView,View}::non_const_data_type``
   * Only relevant for certain external partial specializations of ``{DynRankView, OffsetView, View}`` with the pre Kokkos 5 ``View`` implementation. Equal to ``non_const_data_type`` in most cases.

* ``{DynRankView,OffsetView,View}::array_type``
   * replacement: ``{DynRankView,OffsetView,View}::type``
   * ``array`` is an extremely outdated reference to ``View``.

* ``DynamicView::array_type``
   * replacement: ``DynamicView::uniform_type``
   * consistency with ``View``

* ``ErrorReporter::getCapacity``
   * replacement: ``ErrorReporter::capacity``
   * naming style consistency

* ``ErrorReporter::getNumReports``
   * replacement: ``ErrorReporter::num_reports``
   * naming style consistency

* ``ErrorReporter::getNumReportAttempts``
   * replacement: ``ErrorReporter::num_report_attempts``
   * naming style consistency

* ``ErrorReporter::getReports``
   * replacement: ``ErrorReporter::get_reports``
   * naming style consistency

Deprecated in Kokkos 4.x
===========================

Deprecated in Kokkos 4.7
---------------------------

* ``KOKKOS_MEMORY_ALIGNMENT[_THRESHOLD]``
   * replacement: none
   * Not intended for non-internal use.

* ``Kokkos::MemoryManaged``
   * replacement: none
   * Unneeded due to redundancy with default memory trait and confusing use when requesting unmanaged views with MemoryManaged

* ``KOKKOS_NONTEMPORAL_PREFETCH_{LOAD,STORE}``
   * replacement: none
   * Not intended for non-internal use.

Deprecated in Kokkos 4.6
---------------------------

* ``StaticCrsGraph`` moved to Kokkos Kernels
   * replacement: ``KokkosSparse::StaticCrsGraph``
   * aligns better with functionality provided by ``KokkosKernels``.

* ``native_simd`` and ``native_simd_mask`` types
   * replacement: ``simd`` and ``simd_mask``
   * alignment with the C++ standard

* Makefile support
   * replacement: CMake
   * reducing maintenance burden for a little-used build system

* Direct access to ``d_view`` and ``h_view`` members in ``DualView``
   * replacement: ``view_host()`` and ``view_device()``
   * enforcing invariants in ``DualView``, e.g., consistency between the two ``View`` instances referenced

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

* :cpp:func:`atomic_assign`
   * replacement: :cpp:func:`atomic_store`
   * duplicated functionality

* :cpp:func:`atomic_increment`
   * replacement: :cpp:func:`atomic_inc`
   * duplicated functionality

* :cpp:func:`atomic_decrement`
   * replacement: :cpp:func:`atomic_dec`
   * duplicated functionality

* :cpp:func:`atomic_compare_exchange_strong`
   * replacement: :cpp:func:`atomic_compare_exchange`
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

* ``HIP::detect_device_count()``
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

* static ``ExecutionSpace::concurrency()``
   * replacement: non-static ``ExecutionSpace::concurrency()`` member function
   * concurrency is a property of an execution space instance, not of its type