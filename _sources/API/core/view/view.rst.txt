``View``
========

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

Kokkos View is a potentially reference counted multi dimensional array with compile time layouts and memory space.
Its semantics are similar to that of ``std::shared_ptr``.

Interface
---------

.. code-block:: cpp

    template <class DataType [, class LayoutType] [, class MemorySpace] [, class MemoryTraits]>
    class View;

Parameters
~~~~~~~~~~

.. _LayoutRight: layoutRight.html

.. |LayoutRight| replace:: ``LayoutRight()``

.. _LayoutLeft: layoutLeft.html

.. |LayoutLeft| replace:: ``LayoutLeft()``

.. _LayoutStride: layoutStride.html

.. |LayoutStride| replace:: ``LayoutStride()``

Template parameters other than ``DataType`` are optional, but ordering is enforced.
That means for example that ``LayoutType`` can be omitted but if both ``MemorySpace``
and ``MemoryTraits`` are specified, ``MemorySpace`` must come before ``MemoryTraits``.

* ``DataType``:

  Defines the fundamental scalar type of the ``View`` and its dimensionality.
  The basic structure is ``ScalarType STARS BRACKETS`` where the number of STARS denotes
  the number of runtime length dimensions and the number of BRACKETS defines the compile time dimensions.
  Due to C++ type restrictions runtime dimensions must come first.
  Examples:

  - ``double**``: 2D View of ``double`` with 2 runtime dimensions

  - ``const int***[5][3]``: 5D View of ``int`` with 3 runtime and 2 compile dimensions. The data is ``const``.

  - ``Foo[6][2]``: 2D View of a class ``Foo`` with 2 compile time dimensions.

* ``LayoutType``:

  Determines the mapping of indices into the underlying 1D memory storage.
  Custom Layouts can be implemented, but Kokkos comes with some built-in ones:

  - |LayoutRight|_: strides increase from the right most to the left most dimension. The last dimension has
    a stride of one. This corresponds to how C multi dimensional arrays (``[][][]``) are laid out in memory.

  - |LayoutLeft|_: strides increase from the left most to the right most dimension.
    The first dimension has a stride of one. This is the layout Fortran uses for its arrays.

  - |LayoutStride|_: strides can be arbitrary for each dimension.

* ``MemorySpace``:

  Controls the storage location.
  If omitted the default memory space of the default execution space is used (i.e. ``Kokkos::DefaultExecutionSpace::memory_space``)

.. _Atomic: ../atomics.html

.. |Atomic| replace:: :cppkokkos:func:`Atomic`


* ``MemoryTraits``:

  Sets access properties via enum parameters for the templated ``Kokkos::MemoryTraits<>`` class.
  Possible template parameters are bit-combinations of the following flags:

  - ``Unmanaged``: The View will not be reference counted. The allocation has to be provided to the constructor.

  - |Atomic|_: All accesses to the view will use atomic operations.

  - ``RandomAccess``: Hint that the view is used in a random access manner.
    If the view is also ``const`` this will trigger special load operations on GPUs (i.e. texture fetches).

  - ``Restrict``: There is no aliasing of the view by other data structures in the current scope.

Public Class Members
--------------------

Enums
~~~~~

* ``rank``: rank of the view (i.e. the dimensionality) **(until Kokkos 4.1)**
* ``rank_dynamic``: number of runtime determined dimensions **(until Kokkos 4.1)**
* ``reference_type_is_lvalue_reference``: whether the reference type is a C++ lvalue reference.

**(since Kokkos 4.1)** ``rank`` and ``rank_dynamic`` are static member constants that are convertible to ``size_t``.
Their underlying types are unspecified, but equivalent to ``std::integral_constant`` with a nullary
member function callable from host and device side.
Users are encouraged to use ``rank()`` and ``rank_dynamic()`` (akin to a static member function call)
instead of relying on implicit conversion to an integral type.

The actual type of ``rank[_dynamic]`` as it was defined until Kokkos 4.1 was left up to the implementation
(that is, up to the compiler not to Kokkos) but in practice it was often ``int`` which means
this change may yield warnings about comparing signed and unsigned integral types.
It may also break code that was using the type of ``View::rank``.
Furthermore, it appears that MSVC has issues with the implicit conversion to
``size_t`` in certain constexpr contexts. Calling ``rank()`` or ``rank_dynamic()`` will work in those cases.

Typedefs
~~~~~~~~

.. rubric:: Data Types

.. cpp:type:: data_type

   The ``DataType`` of the View, note ``data_type`` contains the array specifiers (e.g. ``int**[3]``)

.. cpp:type:: const_data_type

   Const version of ``DataType``, same as ``data_type`` if that is already const.

.. cpp:type:: non_const_data_type

   Non-const version of ``DataType``, same as ``data_type`` if that is already non-const.

.. cpp:type:: scalar_array_type

   If ``DataType`` represents some properly specialised array data type such as Sacado FAD types, ``scalar_array_type`` is the underlying fundamental scalar type.

.. cpp:type:: const_scalar_array_type

   Const version of ``scalar_array_type``, same as ``scalar_array_type`` if that is already const

.. cpp:type:: non_const_scalar_array_type

   Non-Const version of ``scalar_array_type``, same as ``scalar_array_type`` if that is already non-const.

.. rubric:: Scalar Types

.. cpp:type:: value_type

   The ``data_type`` stripped of its array specifiers, i.e. the scalar type
   of the data the view is referencing
   (e.g. if ``data_type`` is ``const int**[3]``, ``value_type`` is ``const int``).

.. cpp:type:: const_value_type

   const version of ``value_type``.

.. cpp:type:: non_const_value_type

   non-const version of ``value_type``.

.. rubric:: Spaces

.. cpp:type:: execution_space

   Execution Space associated with the view, will be used for
   performing view initialization, and certain deep_copy operations.

.. cpp:type:: memory_space

   Data storage location type.

.. cpp:type:: device_type

   the compound type defined by ``Device<execution_space,memory_space>``

.. cpp:type:: memory_traits

   The memory traits of the view.

.. cpp:type:: host_mirror_space

   Host accessible memory space used in ``HostMirror``.

.. rubric:: ViewTypes

.. cpp:type:: non_const_type

   this view type with all template parameters explicitly defined.

.. cpp:type:: const_type

   this view type with all template parameters explicitly defined using a ``const`` data type.

.. cpp:type:: HostMirror

   compatible view type with the same ``DataType`` and ``LayoutType`` stored in host accessible memory space.


.. rubric:: Data Handles

.. cpp:type:: reference_type

   return type of the view access operators.

.. cpp:type:: pointer_type

   pointer to scalar type.


.. rubric:: Other

.. cpp:type:: array_layout

   The Layout of the View.

.. cpp:type:: size_type

   index type associated with the memory space of this view.

.. cpp:type:: dimension

   An integer array like type, able to represent the extents of the view.

.. cpp:type:: specialize

   A specialization tag used for partial specialization of the mapping construct underlying a Kokkos View.

Constructors
~~~~~~~~~~~~

.. cppkokkos:function:: View()

   Default Constructor. No allocations are made, no reference counting happens. All extents are zero and its data pointer is NULL.

.. cppkokkos:function:: View( const View<DT, Prop...>& rhs)

   Copy constructor with compatible view. Follows View assignment rules.

.. cppkokkos:function:: View( View&& rhs)

   Move constructor

.. cppkokkos:function:: View( const std::string& name, const IntType& ... indices)

   Standard allocating constructor. The initialization is executed on the default
   instance of the execution space corresponding to ``MemorySpace`` and fences it.

   - ``name``: a user provided label, which is used for profiling and debugging purposes. Names are not required to be unique,

   - ``indices``: Extents of the View.

   - Requires: ``sizeof(IntType...)==rank_dynamic()`` or ``sizeof(IntType...)==rank()``.
     In the latter case, the extents corresponding to compile-time dimensions must match the View type's compile-time extents.

   - Requires: ``array_layout::is_regular == true``.

.. cppkokkos:function:: View( const std::string& name, const array_layout& layout)

   Standard allocating constructor. The initialization is executed on the default
   instance of the execution space corresponding to ``MemorySpace`` and fences it.

   - ``name``: a user provided label, which is used for profiling and debugging purposes.
     Names are not required to be unique,

   - ``layout``: an instance of a layout class. The number of valid extents must
     either match the dynamic rank or the total rank. In the latter case, the extents
     corresponding to compile-time dimensions must match the View type's compile-time extents.

.. cppkokkos:function:: View( const ALLOC_PROP &prop, const IntType& ... indices)

   Allocating constructor with allocation properties (created by a call to `Kokkos::view_alloc`). If an execution space is
   specified in ``prop``, the initialization uses it and does not fence.
   Otherwise, the View is initialized using the default execution space instance corresponding to ``MemorySpace`` and fences it.

   - An allocation properties object is returned by the ``view_alloc`` function.

   - ``indices``: Extents of the View.

   - Requires: ``sizeof(IntType...)==rank_dynamic()`` or ``sizeof(IntType...)==rank()``.
     In the latter case, the extents corresponding to compile-time dimensions must match the View type's compile-time extents.

   - Requires: ``array_layout::is_regular == true``.

.. cppkokkos:function:: View( const ALLOC_PROP &prop, const array_layout& layout)

   Allocating constructor with allocation properties (created by a call to `Kokkos::view_alloc`) and a layout object. If an execution space is
   specified in ``prop``, the initialization uses it and does not fence. Otherwise, the View is
   initialized using the default execution space instance corresponding to ``MemorySpace`` and fences it.

   - An allocation properties object is returned by the ``view_alloc`` function.

   - ``layout``: an instance of a layout class. The number of valid extents must either
     match the dynamic rank or the total rank. In the latter case, the extents corresponding
     to compile-time dimensions must match the View type's compile-time extents.

.. cppkokkos:function:: View( pointer_type ptr, const IntType& ... indices)

   Unmanaged data wrapping constructor.

   - ``ptr``: pointer to a user provided memory allocation. Must provide storage of size ``View::required_allocation_size(n0,...,nR)``

   - ``indices``: Extents of the View.

   - Requires: ``sizeof(IntType...)==rank_dynamic()`` or ``sizeof(IntType...)==rank()``. In the latter case,
     the extents corresponding to compile-time dimensions must match the View type's compile-time extents.

   - Requires: ``array_layout::is_regular == true``.

.. cppkokkos:function:: View( pointer_type ptr, const array_layout& layout)

   Unmanaged data wrapper constructor.

   - ``ptr``: pointer to a user provided memory allocation. Must provide storage
     of size ``View::required_allocation_size(layout)``

   - ``layout``: an instance of a layout class. The number of valid extents must
     either match the dynamic rank or the total rank. In the latter case, the extents
     corresponding to compile-time dimensions must match the View type's compile-time extents.

.. cppkokkos:function:: View( const ScratchSpace& space, const IntType& ... indices)

   Constructor which acquires memory from a Scratch Memory handle.

   - ``space``: scratch memory handle. Typically returned from ``team_handles`` in ``TeamPolicy`` kernels.

   - ``indices``: Runtime dimensions of the view.

   - Requires: ``sizeof(IntType...)==rank_dynamic()`` or ``sizeof(IntType...)==rank()``.
     In the latter case, the extents corresponding to compile-time dimensions must match the View type's compile-time extents.

   - Requires: ``array_layout::is_regular == true``.

.. cppkokkos:function:: View( const ScratchSpace& space, const array_layout& layout)

   Constructor which acquires memory from a Scratch Memory handle.

   - ``space``: scratch memory handle. Typically returned from ``team_handles`` in ``TeamPolicy`` kernels.

   - ``layout``: an instance of a layout class. The number of valid extents must
     either match the dynamic rank or the total rank. In the latter case, the extents
     corresponding to compile-time dimensions must match the View type's compile-time extents.

.. cppkokkos:function:: View( const View<DT, Prop...>& rhs, Args ... args)

   Subview constructor. See ``subview`` function for arguments.

.. cppkokkos:function:: explicit(traits::is_managed) View( const NATURAL_MDSPAN_TYPE& mds )

   :param mds: the mdspan to convert from.

   .. warning::

      :cpp:`explicit(bool)` is only available on C++20 and later. When building Kokkos with C++17, this constructor will be fully implicit.
      Be aware that later upgrading to C++20 will in some cases cause compilation issues in cases where :cpp:`traits::is_managed` is :cpp:`false`.

   :cpp:`NATURAL_MDSPAN_TYPE` is the :ref:`natural mdspan <api-view-natural-mdspans>` of the View. The *natural mdspan* is only available if :cpp:type:`array_layout` is one of :cppkokkos:struct:`LayoutLeft`, :cppkokkos:struct:`LayoutRight`,
   or :cpp:class:`LayoutStride`. This constructor is only available if *natural mdspan* is available.

   Constructs a :cpp:class:`View` by converting from :cpp:any:`mds`. The :cpp:class:`View` will be unmanaged and constructed as if by :cpp:`View(mds.data(), array_layout_from_mapping(mds.mapping()))`

   .. seealso:: :ref:`Natural mdspans <api-view-natural-mdspans>`

   .. versionadded:: 4.4.0

.. cppkokkos:function:: template <class ElementType, class ExtentsType, class LayoutType, class AccessorType> explicit(SEE_BELOW) View(const mdspan<ElementType, ExtentsType, LayoutType, AccessorType>& mds)

   :tparam ElementType: the mdspan element type
   :tparam ExtentsType: the mdspan extents
   :tparam LayoutType: the mdspan layout
   :tparam AccessorType: the mdspan extents

   :param mds: the mdspan to convert from

   .. warning::

      :cpp:`explicit(bool)` is only available on C++20 and later. When building Kokkos with C++17, this constructor will be fully implicit.
      Be aware that later upgrading to C++20 will in some cases cause compilation issues in cases where the condition is false.

   Constructs a :cpp:class:`View` by converting from :cpp:any:`mds`.
   The :cpp:class:`View`'s :ref:`natural mdspan <api-view-natural-mdspans>` must be constructible from :cpp:any:`mds`. The :cpp:class:`View` will be constructed as if by :cpp:`View(NATURAL_MDSPAN_TYPE(mds))`

   In C++20:
      This constructor is implicit if :cpp:any:`mds` is implicitly convertible to the *natural mdspan* of the :cpp:class:`View`.

   .. versionadded:: 4.4.0


Data Access Functions
~~~~~~~~~~~~~~~~~~~~~

.. cppkokkos:function:: reference_type operator() (const IntType& ... indices) const

   Returns a value of ``reference_type`` which may or not be referenceable itself.
   The number of index arguments must match the ``rank`` of the view.
   See notes on ``reference_type`` for properties of the return type.
   Requires: ``sizeof(IntType...)==rank_dynamic()``

.. cppkokkos:function:: reference_type access(const IntType& i0=0, const IntType& i1=0, \
			const IntType& i2=0, const IntType& i3=0, const IntType& i4=0, \
			const IntType& i5=0, const IntType& i6=0, const IntType& i7=0) const

   Returns a value of ``reference_type`` which may or not be referenceable itself.
   The number of index arguments must be equal or larger than the ``rank`` of the view.
   Index arguments beyond ``rank`` must be ``0``, which will be enforced if ``KOKKOS_DEBUG`` is defined.
   See notes on ``reference_type`` for properties of the return type.

Data Layout, Dimensions, Strides
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. cppkokkos:function:: static constexpr size_t rank()

   **since Kokkos 4.1**: Returns the rank of the view.

.. cppkokkos:function:: static constexpr size_t rank_dynamic()

   **since Kokkos 4.1**: Returns the number of runtime determined dimensions.

Note: in practice, ``rank()`` and ``rank_dynamic()`` are not actually
implemented as static member functions but ``rank`` and ``rank_dynamic`` underlying
types have a nullary member function (i.e. callable with no argument).

.. cppkokkos:function:: constexpr array_layout layout() const

   Returns the layout object. Can be used to to construct other views with the same dimensions.

.. cppkokkos:function:: template<class iType> constexpr size_t extent( const iType& dim) const

   Return the extent of the specified dimension. ``iType`` must be an integral type, and ``dim`` must be smaller than ``rank``.

.. cppkokkos:function:: template<class iType> constexpr int extent_int( const iType& dim) const

   Return the extent of the specified dimension as an ``int``. ``iType`` must be an integral type,
   and ``dim`` must be smaller than ``rank``. Compared to ``extent`` this function can be
   useful on architectures where ``int`` operations are more efficient than ``size_t``.
   It also may eliminate the need for type casts in applications which
   otherwise perform all index operations with ``int``.

.. cppkokkos:function:: template<class iType> constexpr size_t stride(const iType& dim) const

   Return the stride of the specified dimension. ``iType`` must be an integral type,
   and ``dim`` must be smaller than ``rank``. Example: ``a.stride(3) == (&a(i0,i1,i2,i3+1,i4)-&a(i0,i1,i2,i3,i4))``

.. cppkokkos:function:: constexpr size_t stride_0() const

   Return the stride of dimension 0.

.. cppkokkos:function:: constexpr size_t stride_1() const

   Return the stride of dimension 1.

.. cppkokkos:function:: constexpr size_t stride_2() const

   Return the stride of dimension 2.

.. cppkokkos:function:: constexpr size_t stride_3() const

   Return the stride of dimension 3.

.. cppkokkos:function:: constexpr size_t stride_4() const

   Return the stride of dimension 4.

.. cppkokkos:function:: constexpr size_t stride_5() const

   Return the stride of dimension 5.

.. cppkokkos:function:: constexpr size_t stride_6() const

   Return the stride of dimension 6.

.. cppkokkos:function:: constexpr size_t stride_7() const

   Return the stride of dimension 7.

.. cppkokkos:function:: template<class iType> void stride(iType* strides) const

   Sets ``strides[r]`` to ``stride(r)`` for all ``r`` with ``0<=r<rank``.
   Sets ``strides[rank]`` to ``span()``. ``iType`` must be an integral type, and ``strides`` must be an array of length ``rank+1``.

.. cppkokkos:function:: constexpr size_t span() const

   Returns the memory span in elements between the element with the
   lowest and the highest address. This can be larger than the product
   of extents due to padding, and or non-contiguous data layout as for example ``LayoutStride`` allows.

.. cppkokkos:function:: constexpr size_t size() const

   Returns the product of extents, i.e. the logical number of elements in the view.

.. cppkokkos:function:: constexpr pointer_type data() const

   Return the pointer to the underlying data allocation.
   WARNING: calling any function that manipulates the behavior
   of the memory (e.g. ``memAdvise``) on memory managed by ``Kokkos`` results in undefined behavior.

.. cppkokkos:function:: bool span_is_contiguous() const

   Whether the span is contiguous (i.e. whether every memory location between
   in span belongs to the index space covered by the view).

.. cppkokkos:function:: static constexpr size_t required_allocation_size(size_t N0=0, size_t N1=0, \
			size_t N2=0, size_t N3=0, \
			size_t N4=0, size_t N5=0, \
			size_t N6=0, size_t N7=0, size_t N8 = 0);

   Returns the number of bytes necessary for an unmanaged view of the provided dimensions. This function is only valid if ``array_layout::is_regular == true``.

.. cppkokkos:function:: static constexpr size_t required_allocation_size(const array_layout& layout);

   Returns the number of bytes necessary for an unmanaged view of the provided layout.

Other
~~~~~

.. cppkokkos:function:: int use_count() const;

   Returns the current reference count of the underlying allocation.

.. cppkokkos:function:: const char* label() const;

   Returns the label of the View.

.. cppkokkos:function:: const bool is_assignable(const View<DT, Prop...>& rhs);

   Returns true if the View can be assigned to rhs.  See below for assignment rules.

.. cppkokkos:function:: void assign_data(pointer_type arg_data);

   Decrement reference count of previously assigned data and set the underlying pointer to arg_data.
   Note that the effective result of this operation is that the view
   is now an unmanaged view; thus, the deallocation of memory associated with
   arg_data is not linked in anyway to the deallocation of the view.

.. cppkokkos:function:: constexpr bool is_allocated() const;

   Returns true if the view points to a valid memory location.
   This function works for both managed and unmanaged views.
   With the unmanaged view, there is no guarantee that referenced
   address is valid, only that it is a non-null pointer.

Conversion to mdspan
~~~~~~~~~~~~~~~~~~~~

.. cppkokkos:function:: template <class OtherElementType, class OtherExtents, class OtherLayoutPolicy, class OtherAccessor> constexpr operator mdspan<OtherElementType, OtherExtents, OtherLayoutPolicy, OtherAccessor>()

   :tparam OtherElementType: the target mdspan element type
   :tparam OtherExtents: the target mdspan extents
   :tparam OtherLayoutPolicy: the target mdspan layout
   :tparam OtherAccessor: the target mdspan accessor

   :constraints: :cpp:class:`View`\ 's :ref:`natural mdspan <api-view-natural-mdspans>` must be assignable to :cpp:`mdspan<OtherElementType, OtherExtents, OtherLayoutPolicy, OtherAccessor>`

   :returns: an mdspan with extents and a layout converted from the :cpp:class:`View`'s *natural mdspan*.

.. cppkokkos:function:: template <class OtherAccessorType = Kokkos::default_accessor<typename traits::value_type>> constexpr auto to_mdspan(const OtherAccessorType& other_accessor = OtherAccessorType{})

   :tparam OtherAccessor: the target mdspan accessor

   :constraints: :cpp:`typename OtherAccessorType::data_handle_type` must be assignable to :cpp:`value_type*`

   :returns: :cpp:class:`View`\ 's :ref:`natural mdspan <api-view-natural-mdspans>`, but with an accessor policy constructed from :cpp:any:`other_accessor`


NonMember Functions
-------------------

.. cppkokkos:function:: template<class ViewDst, class ViewSrc> bool operator==(ViewDst, ViewSrc);

   Returns true if ``value_type``, ``array_layout``, ``memory_space``, ``rank``, ``data()`` and ``extent(r)``, for ``0<=r<rank``, match.

.. cppkokkos:function:: template<class ViewDst, class ViewSrc> bool operator!=(ViewDst, ViewSrc);

   Returns true if any of ``value_type``, ``array_layout``, ``memory_space``, ``rank``, ``data()`` and ``extent(r)``, for ``0<=r<rank`` don't match.

Assignment Rules
----------------

Assignment rules cover the assignment operator as well as copy constructors. We aim at making all logically legal assignments possible,
while intercepting illegal assignments if possible at compile time, otherwise at runtime.
In the following we use ``DstType`` and ``SrcType`` as the type of the destination view and source view respectively.
``dst_view`` and ``src_view`` refer to the runtime instances of the destination and source views, i.e.:

.. code-block:: cpp

    SrcType src_view(...);
    DstType dst_view(src_view);
    dst_view = src_view;

The following conditions must be met at and are evaluated at compile time:

* ``DstType::rank == SrcType::rank``
* ``DstType::non_const_value_type`` is the same as ``SrcType::non_const_value_type``
* If ``std::is_const<SrcType::value_type>::value == true`` than ``std::is_const<DstType::value_type>::value == true``.
* ``MemorySpaceAccess<DstType::memory_space,SrcType::memory_space>::assignable == true``
* If ``DstType::dynamic_rank != DstType::rank`` and ``SrcType::dynamic_rank != SrcType::rank`` then for each dimension ``k`` which is compile time for both it must be true that ``dst_view.extent(k) == src_view.extent(k)``

Additionally the following conditions must be met at runtime:

* If ``DstType::dynamic_rank != DstType::rank`` then for each compile time dimension ``k`` it must be true that ``dst_view.extent(k) == src_view.extent(k)``.

Furthermore there are rules which must be met if ``DstType::array_layout`` is not the same as ``SrcType::array_layout``.
These rules only cover cases where both layouts are one of ``LayoutLeft``, ``LayoutRight`` or ``LayoutStride``

* If neither ``DstType::array_layout`` nor ``SrcType::array_layout`` is ``LayoutStride``:

  - If ``DstType::rank > 1`` then ``DstType::array_layout`` must be the same as ``SrcType::array_layout``.

* If either ``DstType::array_layout`` or ``SrcType::array_layout`` is ``LayoutStride``

  - For each dimension ``k`` it must hold that ``dst_view.extent(k) == src_view.extent(k)``

Assignment Examples
~~~~~~~~~~~~~~~~~~~

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

.. _api-view-natural-mdspans:

Natural mdspans
---------------

.. versionadded:: 4.4.0

C++23 introduces `mdspan <https://en.cppreference.com/w/cpp/container/mdspan>`_, a non-owning multidimensional array view.
:cpp:class:`View` is compatible with :cpp:`std::mdspan` and can be implicitly converted from and to valid mdspans.
These conversion rules are dictated by the *natural mdspan* of a view.
For an mdspan :cpp:`m` of type :cpp:`M` that is the *natural mdspan* of a :cpp:class:`View` :cpp:`v` of type :cpp:`V`, the following properties hold:

#. :cpp:`M::value_type` is :cpp:`V::value_type`
#. :cpp:`M::index_type` is :cpp:`std::size_t`.
#. :cpp:`M::extents_type` is :cpp:`std::extents<M::index_type, Extents...>` where

   * :cpp:`sizeof(Extents...)` is :cpp:`V::rank()`
   * and each element at index :cpp:`r` of :cpp:`Extents...` is :cpp:`V::static_extents(r)` if :cpp:`V::static_extents(r) != 0`, otherwise :cpp:`std::dynamic_extent`

#. :cpp:`M::layout_type` is

   * :cpp:`std::layout_left_padded<std::dynamic_extent>` if :cpp:`V::array_layout` is :cpp:`LayoutLeft`
   * :cpp:`std::layout_right_padded<std::dynamic_extent>` if :cpp:`V::array_layout` is :cpp:`LayoutRight`
   * :cpp:`std::layout_stride` if :cpp:`V::array_layout` is :cpp:any:`LayoutStride`

#. :cpp:`M::accessor_type` is :cpp:`std::default_accessor<V::value_type>`

Additionally, the *natural mdspan* is constructed so that :cpp:`m.data() == v.data()` and for each extent :cpp:`r`, :cpp:`m.extents().extent(r) == v.extent(r)`.

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
