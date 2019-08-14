# `Kokkos::DynamicView`

Header File: `Kokkos_DynamicView.hpp`

Usage: 
Kokkos DyanmicView is a potentially reference counted rank 1 array, without layout, that can be dynamically resized on the host.

  ```c++
  const int chunk_size = 16*1024;
  Kokkos::Experimental::DynamicView<double*> view("v", chunk_size, 10*chunk_size);
  view.resize_serial(3*chunk_size);
  Kokkos::parallel_for("InitializeData", 3*chunk_size, KOKKOS_LAMBDA ( const int i) {
      view(i) = i;
    });
  ```

## Synopsis 

  A rank 1 View-like class, instances can be dynamically resized on the host up to an upper-limit provided by users at construction.

  ```c++
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
  ```
  
## Public Class Members

### Enums

 * `reference_type_is_lvalue_reference`: whether the reference type is a C++ lvalue reference. 

### Typedefs

 * `traits`: `Kokkos::ViewTraits` parent class type.

#### ViewTypes

 * `array_type`: `DynamicView` type templated on `traits::data_type` and `traits::device_type`. 
 * `const_type`: `DynamicView` type templated on `traits::const_data_type` and `traits::device_type`. 
 * `non_const_type`: `DynamicView` type templated on `traits::non_const_data_type` and `traits::device_type`. 
 * `HostMirror`: compatible view type with the same `DataType` and `LayoutType` stored in host accessible memory space. 
  
#### Data Handles

 * `reference_type`: return type of the view access operators.
 * `pointer_type`: pointer to scalar type. 

### Constructors

  * `DynamicView()`: Default Constructor. No allocations are made, no reference counting happens. All extents are zero and its data pointer is NULL.
  * `DynamicView( const DynamicView<RT, RP...>& rhs)`: Copy constructor with compatible view. Follows View assignment rules. 
  * `DynamicView( DynamicView&& rhs)`: Move constructor
  * `DynamicView( const std::string & arg_label, const unsigned min_chunk_size, const unsigned max_extent )`: Standard allocating constructor.
    * `arg_label`: a user provided label, which is used for profiling and debugging purposes. Names are not required to be unique,
    * `min_chunk_size`: user provided minimum chunk size needed for memory allocation, will be raised to nearset power-of-two for more efficient memory access operations.
    * `max_extent`: user provided maximum size, required in order to allocate a chunk-pointer array.
    * The `resize_serial` method must be called after construction to reserve desired amount of memory, bound by `max_extent`.
  
### Data Access Functions
    
  * ```c++
    reference_type operator()( const I0 & i0 , const Args & ... args ) const;
    ```
    Returns a value of `reference_type` which may or not be referenceable itself. The number of index arguments must be 1 (for non-deprecated code).

### Data Resizing, Dimensions, Strides

  * ```c++
    template< typename IntType >
    inline
    typename std::enable_if
      < std::is_integral<IntType>::value &&
        Kokkos::Impl::MemorySpaceAccess< Kokkos::HostSpace
                                       , typename Impl::ChunkArraySpace< typename traits::memory_space >::memory_space 
                                       >::accessible
      >::type
    resize_serial(IntType const & n)
    ```
    Resizes the DynamicView with sufficient chunks of memory of `chunk_size` to store the requested number of elements `n`.
    This method can only be called outside of parallel regions.
    `n` is restricted to be smaller than the `max_extent` value passed to the DynamicView constructor.
    This method must be called after construction of the DynamicView as the constructor sets the requested sizes for `chunk_size` and `max_extent`, but does not take input for actual amount of memory to be used.

  * ```c++
    KOKKOS_INLINE_FUNCTION
    size_t allocation_extent() const noexcept
    ```
    Returns the total size of the product of number of chunks multipled by the chunk size. This may be larger than `size` as this includes total size for total number of complete chunks of memory.
  
  * ```c++
    KOKKOS_INLINE_FUNCTION
    size_t chunk_size() const noexcept
    ```
    Returns the number of entries a chunk of memory may store, always a power of two.
  
  * ```c++
    KOKKOS_INLINE_FUNCTION
    size_t size() const noexcept
    ```
    Returns the number of entries available in the allocation based on the number passed to `resize_serial`. This number is bound by `allocation_extent`.
  
  * ```c++
    template<class iType>
    constexpr size_t extent( const iType& dim) const
    ```
    Return the extent of the specified dimension. `iType` must be an integral type, and `dim` must be smaller than `rank`.
    Returns 1 for rank > 1.
  * ```c++
    template<class iType>
    constexpr int extent_int( const iType& dim) const
    ```
    Return the extent of the specified dimension as an `int`. `iType` must be an integral type, and `dim` must be smaller than `rank`.
    Compared to `extent` this function can be useful on architectures where `int` operations are more efficient than `size_t`. It also may eliminate the need for type casts in applications which otherwise perform all index operations with `int`. 
    Returns 1 for rank > 1.
    
  * ```c++
    template<class iType>
    constexpr size_t stride(const iType& dim) const
    ```
    Return the stride of the specified dimension, always returns 0 for `DynamicView`.
  * ```c++
    constexpr size_t stride_0() const
    ```
    Return the stride of dimension 0, always returns 0 for `DynamicView`s.
  * ```c++
    constexpr size_t stride_1() const
    ```
    Return the stride of dimension 1, always returns 0 for `DynamicView`s.
  * ```c++
    constexpr size_t stride_2() const
    ```
    Return the stride of dimension 2, always returns 0 for `DynamicView`s.
  * ```c++
    constexpr size_t stride_3() const
    ```
    Return the stride of dimension 3, always returns 0 for `DynamicView`s.
  * ```c++
    constexpr size_t stride_4() const
    ```
    Return the stride of dimension 4, always returns 0 for `DynamicView`s.
  * ```c++
    constexpr size_t stride_5() const
    ```
    Return the stride of dimension 5, always returns 0 for `DynamicView`s.
  * ```c++
    constexpr size_t stride_6() const
    ```
    Return the stride of dimension 6, always returns 0 for `DynamicView`s.
  * ```c++
    constexpr size_t stride_7() const
    ```
    Return the stride of dimension 7, always returns 0 for `DynamicView`s.
  * ```c++
    constexpr size_t span() const
    ```
    Always returns 0 for `DynamicView`s.
  * ```c++
    constexpr pointer_type data() const
    ```
    Return the pointer to the underlying data allocation.
  * ```c++
    bool span_is_contiguous() const
    ```
    Whether the span is contiguous, always false for `DynamicView`s.
  
  
### Other
  * ```c++
    int use_count() const
    ```
    Returns the current reference count of the underlying allocation.

  * ```c++
    const char* label() const
    ```
    Returns the label of the DynamicView. 

