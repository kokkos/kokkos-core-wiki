``DynamicView``
===============

.. role:: cpp(code)
   :language: cpp

Header File: ``Kokkos_DynamicView.hpp``

Class Interface
---------------

.. cpp:class:: template<typename DataType , typename ... P> DynamicView

  A potentially reference-counted rank 1 array, without layout, that can be dynamically resized on the host.

  .. rubric:: Public Member Variables

  .. cpp:member:: static constexpr bool reference_type_is_lvalue_reference

    whether the reference type is a C++ lvalue reference.

  .. rubric:: Data Types

  .. cpp:type:: traits

    :cpp:`Kokkos::ViewTraits` parent class type.

  .. rubric:: View Types

  .. cpp:type:: array_type

    :cpp:`DynamicView` type templated on :cpp:`traits::data_type` and :cpp:`traits::device_type`.

  .. cpp:type:: const_type

    :cpp:`DynamicView` type templated on :cpp:`traits::const_data_type` and :cpp:`traits::device_type`.

  .. cpp:type:: non_const_type

    :cpp:`DynamicView` type templated on :cpp:`traits::non_const_data_type` and :cpp:`traits::device_type`.

  .. cpp:type:: HostMirror

    The compatible view type with the same :cpp:`DataType` and :cpp:`LayoutType` stored in host accessible memory space.

  .. rubric:: Data Handle Types

  .. cpp:type:: reference_type

    The return type of the view access operators.

  .. cpp:type:: pointer_type

    The pointer to scalar type.

  .. rubric:: Constructors

  .. cpp:function:: DynamicView()

    The default Constructor. No allocations are made, no reference counting happens. All extents are zero and its data pointer is NULL.

  .. cpp:function:: DynamicView(const DynamicView<RT, RP...>& rhs)

    The copy constructor from a compatible View. Follows View assignment rules.

  .. cpp:function:: DynamicView(DynamicView&& rhs)

    The move constructor

  .. cpp:function:: DynamicView(const std::string & arg_label, const unsigned min_chunk_size, const unsigned max_extent)

    The standard allocating constructor.

    :param arg_label: a user-provided label, which is used for profiling and debugging purposes. Names are not required to be unique
    :param min_chunk_size: a user-provided minimum chunk size needed for memory allocation, will be raised to nearest power-of-two for more efficient memory access operations
    :param max_extent: a user-provided maximum size, required to allocate a chunk-pointer array

    The :any:`resize_serial` method must be called after construction to reserve the desired amount of memory, bound by :any:`max_extent`.

  .. rubric:: Data Access Functions

  .. cpp:function:: reference_type operator() (const I0 & i0 , const Args & ... args) const

    :return: a value of :any:`reference_type` which may or not be referenceable itself. The number of index arguments must be 1 (for non-deprecated code).

  .. rubric:: Data Resizing, Dimensions, Strides

  .. code-block:: cpp

     template< typename IntType >
     inline
     typename std::enable_if
       < std::is_integral<IntType>::value &&
         Kokkos::Impl::MemorySpaceAccess< Kokkos::HostSpace
                                        , typename Impl::ChunkArraySpace< typename traits::memory_space >::memory_space
                                        >::accessible
       >::type
     resize_serial(IntType const & n)

  Resizes the DynamicView with sufficient chunks of memory of :any:`chunk_size` to store the requested number of elements :cpp:`n`.
  This method can only be called outside of parallel regions.
  :cpp:`n` is restricted to be smaller than the :any:`max_extent` value passed to the DynamicView constructor.
  This method must be called after the construction of the DynamicView as the constructor sets the requested sizes for :any:`chunk_size` and :any:`max_extent` , but does not take input for the actual amount of memory to be used.

  .. cpp:function:: size_t allocation_extent() const noexcept

    :return: the total size of the product of the number of chunks multiplied by the chunk size. This may be larger than :any:`size` as this includes the total size for the total number of complete chunks of memory.

  .. cpp:function:: size_t chunk_size() const noexcept

    :return: the number of entries a chunk of memory may store, always a power of two.

  .. cpp:function:: size_t size() const noexcept

    :return: the number of entries available in the allocation based on the number passed to :any:`resize_serial`. This number is bound by :any:`allocation_extent`.

  .. cpp:function:: constexpr size_t extent( const iType& dim) const

    :return: the extent of the specified dimension. :any:`iType` must be an integral type, and :any:`dim` must be smaller than :any:`rank`. Returns 1 for rank > 1.

  .. cpp:function:: constexpr int extent_int( const iType& dim) const

    :return: the extent of the specified dimension as an :any:`int`. :any:`iType` must be an integral type, and :any:`dim` must be smaller than :any:`rank`. Compared to :any:`extent` this function can be useful on architectures where :any:`int` operations are more efficient than :any:`size_t`. It also may eliminate the need for type casts in applications that otherwise perform all index operations with :any:`int`. Returns 1 for rank > 1.

  .. cpp:function:: constexpr size_t stride(const iType& dim) const

    :return: the stride of the specified dimension, always returns 0 for :cpp:`DynamicView`.

  .. cpp:function:: constexpr size_t stride_0() const

    :return: the stride of dimension 0, always returns 0 for :cpp:`DynamicView` s.

  .. cpp:function:: constexpr size_t stride_1() const

    :return: the stride of dimension 1, always returns 0 for :cpp:`DynamicView` s.

  .. cpp:function:: constexpr size_t stride_2() const

    :return: the stride of dimension 2, always returns 0 for :cpp:`DynamicView` s.

  .. cpp:function:: constexpr size_t stride_3() const

    :return: the stride of dimension 3, always returns 0 for :cpp:`DynamicView` s.

  .. cpp:function:: constexpr size_t stride_4() const

    :return: the stride of dimension 4, always returns 0 for :cpp:`DynamicView` s.

  .. cpp:function:: constexpr size_t stride_5() const

    :return: the stride of dimension 5, always returns 0 for :cpp:`DynamicView` s.

  .. cpp:function:: constexpr size_t stride_6() const

    :return: the stride of dimension 6, always returns 0 for :cpp:`DynamicView` s.

  .. cpp:function:: constexpr size_t stride_7() const

    :return: the stride of dimension 7, always returns 0 for :cpp:`DynamicView` s.

  .. cpp:function:: constexpr size_t span() const

    :return: always returns 0 for :cpp:`DynamicView` s.

  .. cpp:function:: constexpr pointer_type data() const

    :return: the pointer to the underlying data allocation.

  .. cpp:function:: bool span_is_contiguous() const

    :return: the span is contiguous, always false for :cpp:`DynamicView` s.

  .. rubric:: Other

  .. cpp:function:: int use_count() const

    :return: the current reference count of the underlying allocation.

  .. cpp:function:: const char* label() const

    :return: the label of the :cpp:`DynamicView`.

  .. cpp:function:: bool is_allocated() const

    :return: true if the View points to a valid set of allocated memory chunks.  Note that this will return false until resize_serial is called with a size greater than 0.



.. code-block:: cpp

   const int chunk_size = 16*1024;
   Kokkos::Experimental::DynamicView<double*> view("v", chunk_size, 10*chunk_size);
   view.resize_serial(3*chunk_size);
   Kokkos::parallel_for("InitializeData", 3*chunk_size, KOKKOS_LAMBDA ( const int i) {
     view(i) = i;
   });


Synopsis
--------

A rank 1 View-like class, instances can be dynamically resized on the host up to an upper limit provided by users at construction.

.. code-block:: cpp

   template< typename DataType , typename ... P >
   class DynamicView : public Kokkos::ViewTraits< DataType , P ... >
   {
   public:
       typedef Kokkos::ViewTraits< DataType , P ... >  traits ;
   private:
       template< class , class ... > friend class DynamicView ;
       typedef Kokkos::Impl::SharedAllocationTracker  track_type ;
       template< class Space , bool = Kokkos::Impl::MemorySpaceAccess< Space , typename traits::memory_space >::accessible > struct verify_space;

       track_type                     m_track ;
       typename traits::value_type ** m_chunks ;      // array of pointers to 'chunks' of memory
       unsigned                       m_chunk_shift ; // ceil(log2(m_chunk_size))
       unsigned                       m_chunk_mask ;  // m_chunk_size - 1
       unsigned                       m_chunk_max ;   // number of entries in the chunk array - each pointing to a chunk of extent == m_chunk_size entries
       unsigned                       m_chunk_size ;  // 2 << (m_chunk_shift - 1)

   public:
       typedef DynamicView< typename traits::data_type, typename traits::device_type > array_type ;
       typedef DynamicView< typename traits::const_data_type, typename traits::device_type > const_type ;
       typedef DynamicView< typename traits::non_const_data_type, typename traits::device_type > non_const_type ;
       typedef DynamicView  HostMirror ;

       typedef Kokkos::Device<typename traits::device_type::execution_space, Kokkos::AnonymousSpace> uniform_device;
       typedef array_type uniform_type;
       typedef const_type uniform_const_type;
       typedef array_type uniform_runtime_type;
       typedef const_type uniform_runtime_const_type;
       typedef DynamicView<typename traits::data_type, uniform_device> uniform_nomemspace_type;
       typedef DynamicView<typename traits::const_data_type, uniform_device> uniform_const_nomemspace_type;
       typedef DynamicView<typename traits::data_type, uniform_device> uniform_runtime_nomemspace_type;
       typedef DynamicView<typename traits::const_data_type, uniform_device> uniform_runtime_const_nomemspace_type;

       enum { Rank = 1 };

       KOKKOS_INLINE_FUNCTION
       size_t allocation_extent() const noexcept;

       KOKKOS_INLINE_FUNCTION
       size_t chunk_size() const noexcept;

       KOKKOS_INLINE_FUNCTION
       size_t size() const noexcept;

       template< typename iType >
       KOKKOS_INLINE_FUNCTION
       size_t extent( const iType & r ) const;

       template< typename iType >
       KOKKOS_INLINE_FUNCTION
       size_t extent_int( const iType & r ) const;

       KOKKOS_INLINE_FUNCTION constexpr size_t stride_0() const;
       KOKKOS_INLINE_FUNCTION constexpr size_t stride_1() const;
       KOKKOS_INLINE_FUNCTION constexpr size_t stride_2() const;
       KOKKOS_INLINE_FUNCTION constexpr size_t stride_3() const;
       KOKKOS_INLINE_FUNCTION constexpr size_t stride_4() const;
       KOKKOS_INLINE_FUNCTION constexpr size_t stride_5() const;
       KOKKOS_INLINE_FUNCTION constexpr size_t stride_6() const;
       KOKKOS_INLINE_FUNCTION constexpr size_t stride_7() const;

       template< typename iType >
       KOKKOS_INLINE_FUNCTION void stride( iType * const s ) const;

       KOKKOS_INLINE_FUNCTION
       int use_count() const;

       inline
       const std::string label() const;

       typedef typename traits::value_type &  reference_type ;
       typedef typename traits::value_type *  pointer_type ;

       enum { reference_type_is_lvalue_reference = std::is_lvalue_reference< reference_type >::value };

       KOKKOS_INLINE_FUNCTION constexpr bool span_is_contiguous() const;
       KOKKOS_INLINE_FUNCTION constexpr size_t span() const;
       KOKKOS_INLINE_FUNCTION constexpr pointer_type data() const;

       template< typename I0 , class ... Args >
       KOKKOS_INLINE_FUNCTION
       reference_type operator()( const I0 & i0 , const Args & ... args ) const;

       template< typename IntType >
       inline
       typename std::enable_if
         < std::is_integral<IntType>::value &&
           Kokkos::Impl::MemorySpaceAccess< Kokkos::HostSpace
                                          , typename Impl::ChunkArraySpace< typename traits::memory_space >::memory_space 
                                          >::accessible
         >::type
       resize_serial( IntType const & n );

       ~DynamicView() = default;
       DynamicView() = default;
       DynamicView( DynamicView && ) = default;
       DynamicView( const DynamicView & ) = default;
       DynamicView & operator = ( DynamicView && ) = default;
       DynamicView & operator = ( const DynamicView & ) = default;

       template< class RT , class ... RP >
       DynamicView( const DynamicView<RT,RP...> & rhs );

       struct Destroy;

       explicit inline
       DynamicView( const std::string & arg_label, const unsigned min_chunk_size, const unsigned max_extent );
   };

