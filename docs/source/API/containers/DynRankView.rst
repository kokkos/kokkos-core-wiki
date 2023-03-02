``DynRankView``
===============

.. role:: cppkokkos(code)
    :language: cppkokkos

Header File: ``<Kokkos_DynRankView.hpp>``

Class Interface
---------------

.. cppkokkos:class:: template <class DataType> DynRankView

.. cppkokkos:class:: template <class LayoutType> DynRankView

.. cppkokkos:class:: template <class MemorySpace> DynRankView

.. cppkokkos:class:: template <class MemoryTraits> DynRankView

    A potentially reference counted multidimensional array with compile time layouts and memory space. Its semantics are similar to that of ``std::shared_ptr``. The ``DynRankView`` differs from the ``Kokkos::View`` in that its rank is not provided with the ``DataType`` template parameter; it is determined dynamically based on the number of extent arguments passed to the constructor. The rank has an upper bound of 7 dimensions.

    Template parameters other than ``DataType`` are optional, but ordering is enforced. That means for example that ``LayoutType`` can be omitted but if both ``MemorySpace`` and ``MemoryTraits`` are specified, ``MemorySpace`` must come before ``MemoryTraits``.

    :tparam DataType: Defines the fundamental scalar type of the ``DynRankView``. The basic structure is ``ScalarType``. Examples:

        * ``double``: a ``DynRankView`` of ``double``, dimensions are passed as arguments to the constructor, the number of which determine the rank.

    :tparam LayoutType: Determines the mapping of indices into the underlying 1D memory storage. Custom Layouts can be implemented, but Kokkos comes with some built-in ones:

        * ``LayoutRight``: Strides increase from the right most to the left most dimension. The last dimension has a stride of one. This corresponds to how C multi dimensional arrays (\ ``[][][]``\ ) are laid out in memory.
        * ``LayoutLeft``: Strides increase from the left most to the right most dimension. The first dimension has a stride of one. This is the layout Fortran uses for its arrays.
        * ``LayoutStride``: Strides can be arbitrary for each dimension.

    :tparam MemorySpace: Controls the storage location. If omitted, the default memory space of the default execution space is used (i.e. ``Kokkos::DefaultExecutionSpace::memory_space``)
    :tparam MemoryTraits: Sets access properties via enum parameters for the templated ``Kokkos::MemoryTraits<>`` class. Enums can be bit combined. Possible values:

        * ``Unmanaged``: The DynRankView will not be reference counted. The allocation has to be provided to the constructor.
        * ``Atomic``: All accesses to the view will use atomic operations.
        * ``RandomAccess``: Hint that the view is used in a random access manner. If the view is also ``const``\ , this will trigger special load operations on GPUs (i.e. texture fetches).
        * ``Restrict``: There is no aliasing of the view by other data structures in the current scope.

    .. rubric:: Public Member Variables

    .. cppkokkos:member:: static constexpr unsigned rank

        The rank of the view (i.e. the dimensionality).

    .. cppkokkos:member:: static constexpr unsigned rank_dynamic

        The number of runtime determined dimensions.

    .. cppkokkos:member:: static constexpr bool reference_type_is_lvalue_reference

        Whether the reference type is a C++ lvalue reference.

    .. rubric:: Data Types

    .. cppkokkos:type:: data_type

        The ``DataType`` of the DynRankView.

    .. cppkokkos:type:: const_data_type

        The const version of ``DataType``, same as ``data_type`` if that is already const.

    .. cppkokkos:type:: non_const_data_type

        The non-const version of ``DataType``, same as ``data_type`` if that is already non-const.

    .. cppkokkos:type:: scalar_array_type

        If ``DataType`` represents some properly specialised array data type such as Sacado FAD types, ``scalar_array_type`` is the underlying fundamental scalar type.

    .. cppkokkos:type:: const_scalar_array_type

        The const version of ``scalar_array_type``, same as ``scalar_array_type`` if that is already const

    .. cppkokkos:type:: non_const_scalar_array_type

        The non-Const version of ``scalar_array_type``, same as ``scalar_array_type`` if that is already non-const.

    .. rubric:: Scalar Types

    .. cppkokkos:type:: value_type

        The ``data_type`` stripped of its array specifiers, i.e. the scalar type of the data the view is referencing (e.g. if ``data_type`` is ``const int*******``, ``value_type`` is ``const int``).

    .. cppkokkos:type:: const_value_type

        The const version of ``value_type``.

    .. cppkokkos:type:: non_const_value_type

        The non-const version of ``value_type``.

    .. rubric:: Spaces

    .. cppkokkos:type:: execution_space

        The Execution Space associated with the view, will be used for performing view initialization, and certain ``deep_copy`` operations.

    .. cppkokkos:type:: memory_space

        The data storage location type.

    .. cppkokkos:type:: device_type

        The compound type defined by ``Device<execution_space,memory_space>``.

    .. cppkokkos:type:: memory_traits

        The memory traits of the view.

    .. cppkokkos:type:: host_mirror_space

        The host accessible memory space used in ``HostMirror``.

    .. rubric:: View Types

    .. cppkokkos:type:: non_const_type

        The view type with all template parameters explicitly defined.

    .. cppkokkos:type:: const_type

        The view type with all template parameters explicitly defined using a ``const`` data type.

    .. cppkokkos:type:: HostMirror

        A compatible view type with the same ``DataType`` and ``LayoutType`` stored in host accessible memory space.

    .. rubric:: Data Handle Types

    .. cppkokkos:type:: reference_type

        The return type of the view access operators.

    .. cppkokkos:type:: pointer_type

        The pointer to scalar type.

    .. rubric:: Other Types

    .. cppkokkos:type:: array_layout

        The layout of the ``DynRankView``.

    .. cppkokkos:type:: size_type

        The index type associated with the memory space of this view.

    .. cppkokkos:type:: dimension

        An integer array like type, able to represent the extents of the view.

    .. cppkokkos:type:: specialize

        A specialization tag used for partial specialization of the mapping construct underlying a Kokkos ``DynRankView``.

    .. rubric:: Constructors

    .. cppkokkos:function:: DynRankView()

        The default constructor. No allocations are made, no reference counting happens. All extents are zero and its data pointer is ``nullptr`` and its rank is set to 0.

    .. cppkokkos:function:: DynRankView(const DynRankView<DT, Prop...>& rhs)

        The copy constructor with compatible DynRankViews. Follows DynRankView assignment rules.

    .. cppkokkos:function:: DynRankView(DynRankView&& rhs)

        The move constructor.

    .. cppkokkos:function:: DynRankView(const View<RT,RP...> & rhs)

        The copy constructor taking View as input.

    .. cppkokkos:function:: DynRankView(const std::string& name, const IntType& ... indices)

        *Requires:* ``array_layout::is_regular == true``

        The standard allocating constructor.

        :param name: a user provided label, which is used for profiling and debugging purposes. Names are not required to be unique
        :param indices: the runtime dimensions of the view

    .. cppkokkos:function:: DynRankView(const std::string& name, const array_layout& layout)

        The standard allocating constructor.

        :param name: a user provided label, which is used for profiling and debugging purposes. Names are not required to be unique
        :param layout: the instance of a layout class

    .. cppkokkos:function:: DynRankView(const AllocProperties& prop, const IntType& ... indices)

        *Requires:* ``array_layout::is_regular == true``

        The allocating constructor with allocation properties. An allocation properties object is returned by the ``view_alloc`` function.

        :param indices: the runtime dimensions of the view

    .. cppkokkos:function:: DynRankView(const AllocProperties& prop, const array_layout& layout)

        The allocating constructor with allocation properties and a layout object.

        :param layout: the instance of a layout class

    .. cppkokkos:function:: DynRankView(const pointer_type& ptr, const IntType& ... indices)

        *Requires:* ``array_layout::is_regular == true``

        The unmanaged data wrapping constructor.

        :param ptr: pointer to a user provided memory allocation. Must provide storage of size ``DynRankView::required_allocation_size(n0,...,nR)``
        :param indices: the runtime dimensions of the view

    .. cppkokkos:function:: DynRankView(const std::string& name, const array_layout& layout)

        The unmanaged data wrapper constructor.

        :param ptr: pointer to a user provided memory allocation. Must provide storage of size ``DynRankView::required_allocation_size(layout)`` (\ *NEEDS TO BE IMPLEMENTED*\ )
        :param layout: the instance of a layout class

    .. cppkokkos:function:: DynRankView( const ScratchSpace& space, const IntType& ... indices)

        The constructor which acquires memory from a Scratch Memory handle.

        *Requires:* ``sizeof(IntType...)==rank_dynamic()`` *and* ``array_layout::is_regular == true``.

        :param space: scratch memory handle. Typically returned from ``team_handles`` in ``TeamPolicy`` kernels
        :param indices: the runtime dimensions of the view

    .. cppkokkos:function:: DynRankView(const ScratchSpace& space, const array_layout& layout)

        The constructor which acquires memory from a Scratch Memory handle.

        :param space: scratch memory handle. Typically returned from ``team_handles`` in ``TeamPolicy`` kernels.
        :param layout: the instance of a layout class

    .. cppkokkos:function:: DynRankView(const DynRankView<DT, Prop...>& rhs, Args ... args)

        The subview constructor. See ``subview`` function for arguments.

    .. rubric:: Data Access Functions

    .. cppkokkos:function:: reference_type operator() (const IntType& ... indices) const

        :return: a value of ``reference_type`` which may or not be reference itself. The number of index arguments must match the ``rank`` of the view. See notes on ``reference_type`` for properties of the return type.

    .. code-block:: cpp
        
        reference_type access (const IntType& i0=0, ... , const IntType& i6=0) const
    
    \
        :return: a value of ``reference_type`` which may or not be reference itself. The number of index arguments must be equal or larger than the ``rank`` of the view. Index arguments beyond ``rank`` must be ``0`` , which will be enforced if ``KOKKOS_DEBUG`` is defined. See notes on ``reference_type`` for properties of the return type.

    .. rubric:: Data Layout, Dimensions, Strides

    .. cppkokkos:function:: constexpr array_layout layout() const

        :return: the layout object. Can be used to to construct other views with the same dimensions.

    .. cppkokkos:function:: template<class iType> constexpr size_t extent( const iType& dim) const

        :return: the extent of the specified dimension. ``iType`` must be an integral type, and ``dim`` must be smaller than ``rank``.

    .. cppkokkos:function:: template<class iType> constexpr int extent_int( const iType& dim) const

        :return: the extent of the specified dimension as an ``int``. ``iType`` must be an integral type, and ``dim`` must be smaller than ``rank``. Compared to ``extent`` this function can be useful on architectures where ``int`` operations are more efficient than ``size_t``. It also may eliminate the need for type casts in applications which otherwise perform all index operations with ``int``.

    .. cppkokkos:function:: template<class iType> constexpr size_t stride(const iType& dim) const

        :return: the stride of the specified dimension. ``iType`` must be an integral type, and ``dim`` must be smaller than ``rank``. Example: ``a.stride(3) == (&a(i0,i1,i2,i3+1,i4)-&a(i0,i1,i2,i3,i4))``

    .. cppkokkos:function:: constexpr size_t stride_0() const

        :return: the stride of dimension 0.

    .. cppkokkos:function:: constexpr size_t stride_1() const

        :return: the stride of dimension 1.

    .. cppkokkos:function:: constexpr size_t stride_2() const

        :return: the stride of dimension 2.

    .. cppkokkos:function:: constexpr size_t stride_3() const

        :return: the stride of dimension 3.

    .. cppkokkos:function:: constexpr size_t stride_4() const

        :return: the stride of dimension 4.

    .. cppkokkos:function:: constexpr size_t stride_5() const

        :return: the stride of dimension 5.

    .. cppkokkos:function:: constexpr size_t stride_6() const

        :return: the stride of dimension 6.

    .. cppkokkos:function:: constexpr size_t stride_7() const

        :return: the stride of dimension 7.

    .. cppkokkos:function:: constexpr size_t span() const

        :return: the memory span in elements between the element with the lowest and the highest address. This can be larger than the product of extents due to padding, and or non-contiguous data layout as for example ``LayoutStride`` allows.

    .. cppkokkos:function:: constexpr pointer_type data() const

        :return: the pointer to the underlying data allocation.

    .. cppkokkos:function:: bool span_is_contiguous() const

        :return: whether the span is contiguous (i.e. whether every memory location between in span belongs to the index space covered by the view).

    .. code-block:: cpp
        
        static constexpr size_t required_allocation_size(size_t N0 = 0, ..., size_t N8 = 0)
    
    \
        :return: the number of bytes necessary for an unmanaged view of the provided dimensions. This function is only valid if ``array_layout::is_regular == true``.

    .. cppkokkos:function:: static constexpr size_t required_allocation_size(const array_layout& layout)

        :return: the number of bytes necessary for an unmanaged view of the provided layout.

    .. rubric:: Other

    .. cppkokkos:function:: int use_count() const

        :return: the current reference count of the underlying allocation.

    .. cppkokkos:function:: const char* label() const;

        :return: the label of the ``DynRankView``.

    .. cppkokkos:function:: constexpr unsigned rank() const

        :return: the dynamic rank of the ``DynRankView``

    .. cppkokkos:function:: constexpr bool is_allocated() const

        :return: true if the view points to a valid memory location. This function works for both managed and unmanaged views. With the unmanaged view, there is no guarantee that referenced address is valid, only that it is a non-null pointer.

Assignment Rules
----------------

Assignment rules cover the assignment operator as well as copy constructors. We aim at making all logically legal assignments possible, while intercepting illegal assignments if possible at compile time, otherwise at runtime. In the following, we use ``DstType`` and ``SrcType`` as the type of the destination view and source view respectively. ``dst_view`` and ``src_view`` refer to the runtime instances of the destination and source views, i.e.:

.. code-block:: cpp

    ScrType src_view(...);
    DstType dst_view(src_view);
    dst_view = src_view;

The following conditions must be met at and are evaluated at compile time:

* ``DstType::rank == SrcType::rank``
* ``DstType::non_const_value_type`` is the same as ``SrcType::non_const_value_type``
* If ``std::is_const<SrcType::value_type>::value == true`` than ``std::is_const<DstType::value_type>::value == true``.
* ``MemorySpaceAccess<DstType::memory_space,SrcType::memory_space>::assignable == true``

Furthermore there are rules which must be met if ``DstType::array_layout`` is not the same as ``SrcType::array_layout``. These rules only cover cases where both layouts are one of ``LayoutLeft`` , ``LayoutRight`` or ``LayoutStride``

* If neither ``DstType::array_layout`` nor ``SrcType::array_layout`` is ``LayoutStride``:
    - If ``DstType::rank > 1`` than ``DstType::array_layout`` must be the same as ``SrcType::array_layout``.

* If either ``DstType::array_layout`` or ``SrcType::array_layout`` is ``LayoutStride``
    - For each dimension ``k`` it must hold that ``dst_view.extent(k) == src_view.extent(k)``

Examples
--------

.. code-block:: cpp

    #include<Kokkos_Core.hpp>
    #include<cstdio> 

    int main(int argc, char* argv[]) {
        Kokkos::initialize(argc,argv);

        int N0 = atoi(argv[1]);
        int N1 = atoi(argv[2]);

        Kokkos::DynRankView<double> a("A",N0);
        Kokkos::DynRankView<double> b("B",N1);

        Kokkos::parallel_for("InitA", N0, KOKKOS_LAMBDA (const int& i) {
            a(i) = i;
        });

        Kokkos::parallel_for("InitB", N1, KOKKOS_LAMBDA (const int& i) {
            b(i) = i;
        });

        Kokkos::DynRankView<double,Kokkos::LayoutLeft> c("C",N0,N1);
        {
            Kokkos::DynRankView<const double> const_a(a);
            Kokkos::DynRankView<const double> const_b(b);
            Kokkos::parallel_for("SetC", Kokkos::MDRangePolicy<Kokkos::Rank<2,Kokkos::Iterate::Left>>({0,0},{N0,N1}),
                KOKKOS_LAMBDA (const int& i0, const int& i1) {
                c(i0,i1) = a(i0) * b(i1);
            });
        }

        Kokkos::finalize();
    }
