``Kokkos::View``
================

.. role:: cpp(code)
   :language: cpp

Header File: ``Kokkos_Core.hpp``

Class Interface
---------------

.. cpp:class:: template <class DataType, class... Traits> View

  A potentially reference counted multi dimensional array with compile time layouts and memory space.
  Its semantics are similar to that of :cpp:`std::shared_ptr`.

  Template parameters other than :any:`DataType` are optional, but ordering is enforced. That means for example that :cpp:`LayoutType` can be omitted but if both :cpp:`MemorySpace` and :cpp:`MemoryTraits` are specified, :cpp:`MemorySpace` must come before :cpp:`MemoryTraits`.

  :tparam DataType: Defines the fundamental scalar type of the :any:`View` and its dimensionality. The basic structure is ``ScalarType STARS BRACKETS`` where the number of STARS denotes the number of runtime length dimensions and the number of BRACKETS defines the compile time dimensions. Due to C++ type restrictions runtime dimensions must come first. Examples:

    * :cpp:`double**`: 2D View of :cpp:`double` with 2 runtime dimensions
    * :cpp:`const int***[5][3]`: 5D View of :cpp:`int` with 3 runtime and 2 compile dimensions. The data is :cpp:`const`.
    * :cpp:`Foo[6][2]`: 2D View of a class :cpp:`Foo` with 2 compile time dimensions.

  :tparam LayoutType: Determines the mapping of indices into the underlying 1D memory storage. Custom Layouts can be implemented, but Kokkos comes with some built-in ones:

    * `LayoutRight <layoutRight.html>`_: strides increase from the right most to the left most dimension. The last dimension has a stride of one. This corresponds to how C multi dimensional arrays (``[][][]``) are laid out in memory.
    * `LayoutLeft <layoutLeft.html>`_: strides increase from the left most to the right most dimension. The first dimension has a stride of one. This is the layout Fortran uses for its arrays.
    * `LayoutStride <layoutStride.html>`_: strides can be arbitrary for each dimension.

  :tparam MemorySpace: Controls the storage location. If omitted the default memory space of the default execution space is used (i.e. :cpp:`Kokkos::DefaultExecutionSpace::memory_space`)
  :tparam MemoryTraits: Sets access properties via enum parameters for the templated :cpp:`Kokkos::MemoryTraits<>` class. Enums can be combined bit combined. Posible values:

    * :cpp:`Unmanaged`: The View will not be reference counted. The allocation has to be provided to the constructor.
    * `Atomic <../atomics.html>`_: All accesses to the view will use atomic operations.
    * :cpp:`RandomAccess`: Hint that the view is used in a random access manner. If the view is also :cpp:`const` this will trigger special load operations on GPUs (i.e. texture fetches).
    * :cpp:`Restrict`: There is no aliasing of the view by other data structures in the current scope.

  .. rubric:: Public Member Variables

  .. cpp:member:: static constexpr unsigned rank

    the rank of the view (i.e. the dimensionality).

  .. cpp:member:: static constexpr unsigned rank_dynamic

    the number of runtime determined dimensions.

  .. cpp:member:: static constexpr bool reference_type_is_lvalue_reference

    whether the reference type is a C++ lvalue reference.

  .. rubric:: Data Types

  .. cpp:type:: data_type

    The ``DataType`` of the View, note ``data_type`` contains the array specifiers (e.g. :cpp:`int**[3]`)

  .. cpp:type:: const_data_type

    The const version of ``DataType``, same as ``data_type`` if that is already const.

  .. cpp:type:: non_const_data_type

    The non-const version of ``DataType``, same as ``data_type`` if that is already non-const.

  .. cpp:type:: scalar_array_type

    If ``DataType`` represents some properly specialised array data type such as Sacado FAD types, ``scalar_array_type`` is the underlying fundamental scalar type.

  .. cpp:type:: const_scalar_array_type

    The const version of ``scalar_array_type``, same as ``scalar_array_type`` if that is already const

  .. cpp:type:: non_const_scalar_array_type

    The non-Const version of ``scalar_array_type``, same as ``scalar_array_type`` if that is already non-const.

  .. rubric:: Scalar Types

  .. cpp:type:: value_type

    The ``data_type`` stripped of its array specifiers, i.e. the scalar type of the data the view is referencing (e.g. if ``data_type`` is :cpp:`const int**[3]`, ``value_type`` is ``const int``.

  .. cpp:type:: const_value_type

    The const version of ``value_type``.

  .. cpp:type:: non_const_value_type

    The non-const version of ``value_type``.

  .. rubric:: Spaces

  .. cpp:type:: execution_space

    The Execution Space associated with the view, will be used for performing view initialization, and certain ``deep_copy`` operations.

  .. cpp:type:: memory_space

    The data storage location type.

  .. cpp:type:: device_type

    The compound type defined by :cpp:`Device<execution_space,memory_space>`

  .. cpp:type:: memory_traits

    The memory traits of the view.

  .. cpp:type:: host_mirror_space

    The host accessible memory space used in `HostMirror`.

  .. rubric:: View Types

  .. cpp:type:: non_const_type

    The view type with all template parameters explicitly defined.

  .. cpp:type:: const_type

    The view type with all template parameters explicitly defined using a :cpp:`const` data type.

  .. cpp:type:: HostMirror

    A compatible view type with the same ``DataType`` and ``LayoutType`` stored in host accessible memory space.

  .. rubric:: Data Handle Types

  .. cpp:type:: reference_type

    The return type of the view access operators.

  .. cpp:type:: pointer_type

    The pointer to scalar type.

  .. rubric:: Other Types

  .. cpp:type:: array_layout

    The Layout of the View.

  .. cpp:type:: size_type

    The index type associated with the memory space of this view.

  .. cpp:type:: dimension

    An integer array like type, able to represent the extents of the view.

  .. cpp:type:: specialize

    A specialization tag used for partial specialization of the mapping construct underlying a Kokkos View.

  .. rubric:: Constructors

  .. cpp:function:: View()

    The default constructor. No allocations are made, no reference counting happens. All extents are zero and its data pointer is :cpp:`nullptr`.

  .. cpp:function:: View(const View<DT, Prop...>& rhs)

    The copy constructor with compatible view. Follows View assignment :ref:`rules <Assignment Rules>`.

  .. cpp:function:: View(View&& rhs)

    The move constructor.

  .. cpp:function:: View(const std::string& name, const IntType& ... indices)

    *Requires:* :cpp:`sizeof(IntType...)==rank_dynamic()` *and*  :cpp:`array_layout::is_regular == true`.

    Standard allocating constructor. The initialization is executed on the default instance of the execution space corresponding to ``MemorySpace`` and fences it.

    :param name: a user provided label, which is used for profiling and debugging purposes. Names are not required to be unique.
    :param indices: the runtime dimensions of the view.

  .. cpp:function:: View(const std::string& name, const array_layout& layout)

    Standard allocating constructor. The initialization is executed on the default instance of the execution space corresponding to ``MemorySpace`` and fences it.

    :param name: a user provided label, which is used for profiling and debugging purposes. Names are not required to be unique.
    :param layout: an instance of a layout class.

  .. cpp:function:: View(const AllocProperties& prop, const IntType& ... indices)

    *Requires:* :cpp:`sizeof(IntType...)==rank_dynamic()` *and*  :cpp:`array_layout::is_regular == true`.

    Allocating constructor with allocation properties. If an execution space is specified in ``prop``, the initialization uses it and does not fence. Otherwise, the View is initialized using the default execution space instance corresponding to ``MemorySpace`` and fences it.

    :param prop: an allocation properties object returned by the `view_alloc` function.
    :param indices: the runtime dimensions of the view.

  .. cpp:function:: View( const AllocProperties& prop, const array_layout& layout)

    Allocating constructor with allocation properties and a layout object. If an execution space is specified in ``prop``, the initialization uses it and does not fence. Otherwise, the View is initialized using the default execution space instance corresponding to ``MemorySpace`` and fences it.

    :param prop: an allocation properties object returned by the `view_alloc` function.
    :param layout: an instance of a layout class.

  .. cpp:function:: View( const pointer_type& ptr, const IntType& ... indices)

    *Requires:* :cpp:`sizeof(IntType...)==rank_dynamic()` *and*  :cpp:`array_layout::is_regular == true`.

    An unmanaged data wrapping constructor.

    :param ptr: a pointer to a user provided memory allocation. Must provide storage of size :func:`View::required_allocation_size(n0,...,nR)`
    :param indices: the runtime dimensions of the view.

  .. cpp:function:: View( const pointer_type& ptr, const array_layout& layout)

    An unmanaged data wrapper constructor.

    :param ptr: a pointer to a user provided memory allocation. Must provide storage of size :func:`View::required_allocation_size(n0,...,nR)` (*NEEDS TO BE IMPLEMENTED*)
    :param layout: an instance of a layout class.

  .. cpp:function:: View( const ScratchSpace& space, const IntType& ... indices)

    *Requires:* :cpp:`sizeof(IntType...)==rank_dynamic()` *and*  :cpp:`array_layout::is_regular == true`.

    A constructor which acquires memory from a Scratch Memory handle.

    :param space: a scratch memory handle. Typically returned from :cpp:`team_handles` in :cpp:`TeamPolicy` kernels.
    :param indices: the runtime dimensions of the view.

  .. cpp:function:: View( const ScratchSpace& space, const array_layout& layout)

    A constructor which acquires memory from a Scratch Memory handle.

    :param space: a scratch memory handle. Typically returned from :cpp:`team_handles` in :cpp:`TeamPolicy` kernels.
    :param layout: an instance of a layout class.

  .. cpp:function:: View( const View<DT, Prop...>& rhs, Args ... args)

    The subview constructor.

    .. seealso::

      The :func:`subview` free function.

  .. ...........................................................................

  .. rubric:: Data Access Functions

  .. cpp:function:: reference_type operator() (const IntType& ... indices) const

    *Requires:* :cpp:`sizeof(IntType...)==rank_dynamic()`.

    :param indices: The index to access the view at. The number of index arguments must match the :any:`rank` of the view.
    :return: a value of :any:`reference_type` which may or not be referenceable itself.

  .. cpp:function:: reference_type access(const IntType& i0=0, const IntType& i1=0, const IntType& i2=0, const IntType& i3=0, const IntType& i4=0, const IntType& i5=0, const IntType& i6=0, const IntType& i7=0)

    :param i...: The index arguments to access the view at. Index arguments beyond :any:`rank` must be ``0``, which will be enforced if :c:macro:`KOKKOS_DEBUG` is defined.

    :return: a value of `reference_type` which may or not be referenceable itself.

  .. rubric:: Data Layout, Dimensions, Strides

  .. cpp:function:: constexpr array_layout layout() const

    :return: the layout object that be used to to construct other views with the same dimensions.

  .. cpp:function:: template<class iType> constexpr size_t extent( const iType& dim) const

    *Requires:* :any:`iType` *must be integral.*

    *Preconditions:* :any:`dim` *must be less than* :any:`rank`.

    :return: the extent of the specified dimension.

  .. cpp:function:: template<class iType> constexpr int extent_int( const iType& dim) const

    *Requires:* :any:`iType` *must be integral.*

    *Preconditions:* :any:`dim` *must be less than* :any:`rank`.

    :return: the extent of the specified dimension as an :cpp:`int`.

    Compared to :cpp:`extent` this function can be useful on architectures where :cpp:`int` operations are more efficient than :cpp:`size_t`. It also may eliminate the need for type casts in applications which otherwise perform all index operations with :cpp:`int`.

  .. cpp:function:: template<class iType> constexpr size_t stride(const iType& dim) const

    *Requires:* :any:`iType` *must be integral.*

    *Preconditions:* :any:`dim` *must be less than* :any:`rank`.

    :return: the stride of the specified dimension.

    .. code-block:: cpp
      :caption: Example of using :any:`stride`.

      a.stride(3) == (&a(i0,i1,i2,i3+1,i4)-&a(i0,i1,i2,i3,i4))

  .. cpp:function:: constexpr size_t stride_0() const

    :return: the stride of dimension 0.

  .. cpp:function:: constexpr size_t stride_1() const

    :return: the stride of dimension 1.

  .. cpp:function:: constexpr size_t stride_2() const

    :return: the stride of dimension 2.

  .. cpp:function:: constexpr size_t stride_3() const

    :return: the stride of dimension 3.

  .. cpp:function:: constexpr size_t stride_4() const

    :return: the stride of dimension 4.

  .. cpp:function:: constexpr size_t stride_5() const

    :return: the stride of dimension 5.

  .. cpp:function:: constexpr size_t stride_6() const

    :return: the stride of dimension 6.

  .. cpp:function:: constexpr size_t stride_7() const

    :return: the stride of dimension 7.

  .. cpp:function:: template<class iType> void stride(iType* strides) const

    *Requires:* :any:`iType` *must be integral.*

    *Preconditions:* :any:`strides` *must be an array of length* :any:`rank` + 1.

    :param strides: an array of length :any:`rank` + 1 that will be used to store the stride

    Sets ``strides[r]`` to ``stride(r)`` for all ``r`` with :cpp:`0<=r<rank`. Sets ``strides[rank]`` to ``span()``.

  .. cpp:function:: constexpr size_t span() const

    :return: the memory span in elements between the element with the lowest and the highest address. This can be larger than the product of extents due to padding, and or non-contiguous data layout as for example `LayoutStride` allows.

  .. cpp:function:: constexpr size_t size() const

    :return: the product of extents, i.e. the logical number of elements in the view.

  .. cpp:function:: constexpr pointer_type data() const

    :return: the pointer to the underlying data allocation.

  .. cpp:function:: bool span_is_contiguous() const

    :return: whether the span is contiguous (i.e. whether every memory location between in span belongs to the index space covered by the view).

  .. cpp:function:: static constexpr size_t required_allocation_size(size_t N0 = 0, size_t N1 = 0, size_t N2 = 0, size_t N3 = 0, size_t N4 = 0, size_t N5 = 0, size_t N6 = 6, size_t N7 = 0, size_t N8 = 0);

    *Preconditions:* `array_layout::is_regular` *is true.*

    :param N...: the dimensions of the intended unmanaged :any:`View`
    :return: the number of bytes necessary for an unmanaged :any:`View` of the provided dimensions.

  .. cpp:function:: static constexpr size_t required_allocation_size(const array_layout& layout);

    :layout: the requested layout
    :return: the number of bytes necessary for an unmanaged view of the provided layout.

  .. rubric:: Other

  .. cpp:function:: int use_count() const

    :return: the current reference count of the underlying allocation.

  .. cpp:function:: const char* label() const;

    :return: the label of the :any:`View`.

  .. cpp:function:: const bool is_assignable(const View<DT, Prop...>& rhs);

    :return: true if the View can be assigned to rhs.

    .. seealso::
      :ref:`Assignment Rules`

  .. cpp:function:: void assign_data(pointer_type arg_data);

    :param arg_data: the new data pointer

    Decrement reference count of previously assigned data and set the underlying pointer to arg_data.  Note that the effective result of this operation is that the view is now an unmanaged view; thus, the deallocation of memory associated with arg_data is not linked in anyway to the deallocation of the view.

  .. cpp:function:: constexpr bool is_allocated() const

    :return: true if the view points to a valid memory location.  This function works for both managed and unmanaged views. With the unmanaged view, there is no guarantee that referenced address is valid, only that it is a non-null pointer.

Non-Member Functions
--------------------

.. cpp:function:: template<class ViewDst, class ViewSrc> bool operator==(ViewDst, ViewSrc);

  :tparam ViewDst: the first view type
  :tparam ViewSrc: the second view type

  :return: true if :cpp:type:`~View::value_type`, :cpp:type:`~View::array_layout`, :cpp:any:`~View::memory_space`, :cpp:any:`~View::rank`, :cpp:any:`~View::data()` and :cpp:any:`~View::extent` (r), for :cpp:`0<=r<rank`, match.

.. cpp:function:: template<class ViewDst, class ViewSrc> bool operator!=(ViewDst, ViewSrc);

  :tparam ViewDst: the first view type
  :tparam ViewSrc: the second view type

  :return: true if :cpp:type:`~View::value_type`, :cpp:type:`~View::array_layout`, :cpp:any:`~View::memory_space`, :cpp:any:`~View::rank`, :cpp:any:`~View::data()` and :cpp:any:`~View::extent` (r), for :cpp:`0<=r<rank`, do not match.

.. _Assignment Rules:

Assignment Rules
----------------

Assignment rules cover the assignment operator as well as copy constructors. We aim at making all logically legal assignments possible, 
while intercepting illegal assignments if possible at compile time, otherwise at runtime.
In the following we use ``DstType`` and ``SrcType`` as the type of the destination view and source view respectively. 
``dst_view`` and ``src_view`` refer to the runtime instances of the destination and source views, i.e.:

.. code-block:: cpp

   ScrType src_view(...);
   DstType dst_view(src_view);
   dst_view = src_view;

The following conditions must be met at and are evaluated at compile time:

  * :cpp:`DstType::rank == SrcType::rank`
  * :cpp:`DstType::non_const_value_type` is the same as :cpp:`SrcType::non_const_value_type`
  * If :cpp:`std::is_const<SrcType::value_type>::value == true` than :cpp:`std::is_const<DstType::value_type>::value == true`.
  * :cpp:`MemorySpaceAccess<DstType::memory_space,SrcType::memory_space>::assignable == true`
  * If :cpp:`DstType::dynamic_rank != DstType::rank` and :cpp:`SrcType::dynamic_rank != SrcType::rank` than for each dimension :cpp:`k` which is compile time for both it must be true that :cpp:`dst_view.extent(k) == src_view.extent(k)`

Additionally the following conditions must be met at runtime:

 * If :cpp:`DstType::dynamic_rank != DstType::rank` than for each compile time dimension :cpp:`k` it must be true that :cpp:`dst_view.extent(k) == src_view.extent(k)`.

Furthermore there are rules which must be met if :cpp:`DstType::array_layout` is not the same as :cpp:`SrcType::array_layout`.
These rules only cover cases where both layouts are one of :cpp:`LayoutLeft`, :cpp:`LayoutRight` or :cpp:`LayoutStride`

 * If neither :cpp:`DstType::array_layout` nor :cpp:`SrcType::array_layout` is `LayoutStride`:

   * If :cpp:`DstType::rank > 1` than :cpp:`DstType::array_layout` must be the same as :cpp:`SrcType::array_layout`.

 * If either :cpp:`DstType::array_layout` or :cpp:`SrcType::array_layout` is :cpp:`LayoutStride`

   * For each dimension :cpp:`k` it must hold that :cpp:`dst_view.extent(k) == src_view.extent(k)`

Assignment Examples
^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

     View<int*>       a1 = View<int*>("A1",N);     // OK
     View<int**>      a2 = View<int*[10]>("A2",N); // OK
     View<int*[10]>   a3 = View<int**>("A3",N,M);  // OK if M == 10 otherwise runtime failure
     View<const int*> a4 = a1;                     // OK
     View<int*>       a5 = a4;                     // Error: const to non-const assignment
     View<int**>      a6 = a1;                     // Error: Ranks do not match
     View<int*[8]>    a7 = a3;                     // Error: compile time dimensions do not match
     View<int[4][10]> a8 = a3;                     // OK if N == 4 otherwise runtime failure
     View<int*, LayoutLeft>    a9  = a1;           // OK since a1 is either LayoutLeft or LayoutRight
     View<int**, LayoutStride> a10 = a8;           // OK
     View<int**>               a11 = a10;          // OK
     View<int*, HostSpace> a12 = View<int*, CudaSpace>("A12",N); // Error: non-assignable memory spaces
     View<int*, HostSpace> a13 = View<int*, CudaHostPinnedSpace>("A13",N); // OK

Examples
--------

.. code-block:: cpp

   #include<Kokkos_Core.hpp>
   #include<cstdio> 

   int main(int argc, char* argv[]) {
      Kokkos::initialize(argc,argv);

      int N0 = atoi(argv[1]);
      int N1 = atoi(argv[2]);

      Kokkos::View<double*> a("A",N0);
      Kokkos::View<double*> b("B",N1);

      Kokkos::parallel_for("InitA", N0, KOKKOS_LAMBDA (const int& i) {
        a(i) = i;
      });

      Kokkos::parallel_for("InitB", N1, KOKKOS_LAMBDA (const int& i) {
        b(i) = i;
      });

      Kokkos::View<double**,Kokkos::LayoutLeft> c("C",N0,N1);
      {
        Kokkos::View<const double*> const_a(a);
        Kokkos::View<const double*> const_b(b);
        Kokkos::parallel_for("SetC", Kokkos::MDRangePolicy<Kokkos::Rank<2,Kokkos::Iterate::Left>>({0,0},{N0,N1}),
          KOKKOS_LAMBDA (const int& i0, const int& i1) {
          c(i0,i1) = a(i0) * b(i1);
        });
      }

      Kokkos::finalize();
   }
