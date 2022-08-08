# `DualView`

Container to manage mirroring a Kokkos::View that references device memory with a Kokkos::View that references host memory.  The class provides capabilities to manage data which exists in two different memory spaces at the same time.  It supports views with the same layout on two memory spaces as well as modified flags for both allocations.  Users are responsible for updating the modified flags manually if they change the data in either memory space, by calling the sync() method, which is templated on the device with the modified data.  Users may also synchronize data by calling the modify() function, which is templated on the device that requires synchronization (i.e., the target of the one-way copy operation).
 
The DualView class also provides convenience methods such as realloc, resize and capacity which call the appropriate methods of the underlying [Kokkos::View](../core/view/view) objects.
 
The four template arguments are the same as those of Kokkos::View.
 
 - DataType, The type of the entries stored in the container.
 
 - Layout, The array's layout in memory.
 
 - Device, The Kokkos Device type.  If its memory space is not the same as the host's memory space, then DualView will contain two separate Views: one in device memory, and one in host memory.  Otherwise, DualView will only store one View.
 
 - MemoryTraits (optional) The user's intended memory access behavior.  Please see the documentation of Kokkos::View for examples.  The default suffices for most users.

Usage:

```c++
using view_type = Kokkos::DualView<Scalar**, 
                                   Kokkos::LayoutLeft, 
                                   Device>
view_type a("A", n, m);

Kokkos::deep_copy(a.d_view, 1);
a.template modify<typename view_type::execution_space>();
a.template sync<typename view_type::host_mirror_space>();

Kokkos::deep_copy(a.h_view, 2);
a.template modify<typename ViewType::host_mirror_space>();
a.template sync<typename ViewType::execution_space>();
```

## Synopsis

```C++
template <class DataType, class Arg1Type = void, class Arg2Type = void,
          class Arg3Type = void>
class DualView : public ViewTraits<DataType, Arg1Type, Arg2Type, Arg3Type> {

 public:
  // Typedefs for device types and various Kokkos::View specializations.
  typedef ViewTraits<DataType, Arg1Type, Arg2Type, Arg3Type> traits;

  // The Kokkos Host Device type;
  typedef typename traits::host_mirror_space host_mirror_space;

  // The type of a Kokkos::View on the device.
  typedef View<typename traits::data_type, Arg1Type, Arg2Type, Arg3Type> t_dev;

  // The type of a Kokkos::View host mirror of `t_dev`.
  typedef typename t_dev::HostMirror t_host;

  // The type of a const View on the device.
  typedef View<typename traits::const_data_type, Arg1Type, Arg2Type, Arg3Type>
      t_dev_const;

  // The type of a const View host mirror of `t_dev_const`.
  typedef typename t_dev_const::HostMirror t_host_const;

  // The type of a const, random-access View on the device.
  typedef View<typename traits::const_data_type, typename traits::array_layout,
               typename traits::device_type,
               Kokkos::MemoryTraits<Kokkos::RandomAccess> >
      t_dev_const_randomread;

  //The type of a const, random-access View host mirror of
  //`t_dev_const_randomread`.
  typedef typename t_dev_const_randomread::HostMirror t_host_const_randomread;

  // The type of an unmanaged View on the device.
  typedef View<typename traits::data_type, typename traits::array_layout,
               typename traits::device_type, MemoryUnmanaged>
      t_dev_um;

  // The type of an unmanaged View host mirror of \c t_dev_um.
  typedef View<typename t_host::data_type, typename t_host::array_layout,
               typename t_host::device_type, MemoryUnmanaged>
      t_host_um;

  // The type of a const unmanaged View on the device.
  typedef View<typename traits::const_data_type, typename traits::array_layout,
               typename traits::device_type, MemoryUnmanaged>
      t_dev_const_um;

  // The type of a const unmanaged View host mirror of \c t_dev_const_um.
  typedef View<typename t_host::const_data_type, typename t_host::array_layout,
               typename t_host::device_type, MemoryUnmanaged>
      t_host_const_um;

  // The type of a const, random-access View on the device.
  typedef View<typename t_host::const_data_type, typename t_host::array_layout,
               typename t_host::device_type,
               Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >
      t_dev_const_randomread_um;

  //The type of a const, random-access View host mirror of
  //`t_dev_const_randomread`.
  typedef
      typename t_dev_const_randomread::HostMirror t_host_const_randomread_um;

  // The two View instances.
  t_dev d_view;
  t_host h_view;

  // Counters to keep track of changes ("modified" flags)

  typedef View<unsigned int[2], LayoutLeft, typename t_host::execution_space>
      t_modified_flags;
  typedef View<unsigned int, LayoutLeft, typename t_host::execution_space>
      t_modified_flag;
  t_modified_flags modified_flags;
  t_modified_flag modified_host, modified_device;

  // Constructors

  // Empty constructor.
  //
  // Both device and host View objects are constructed using their
  // default constructors.  The "modified" flags are both initialized
  // to "unmodified."
  DualView();

  // Constructor that allocates View objects on both host and device.
  //
  // This constructor works like the analogous constructor of View.
  // The first argument is a string label, which is entirely for your
  // benefit.  (Different DualView objects may have the same label if
  // you like.)  The arguments that follow are the dimensions of the
  // View objects.  For example, if the View has three dimensions,
  // the first three integer arguments will be nonzero, and you may
  // omit the integer arguments that follow.
  DualView(const std::string& label,
           const size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
           const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
           const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
           const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
           const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
           const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
           const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
           const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG);

                                                                                                                                                                                                                                             
  /// Constructor that allocates View objects on both host and device.                                                                                                                                                                
  ///                                                                                                                                                                                                                                        
  /// This constructor works like the analogous constructor of View.                                                                                                                                                                         
  /// The first arguments are wrapped up in a ViewCtor class, this allows                                                                                                                                                                    
  /// for a label, without initializing, and all of the other things that can                                                                                                                                                                
  /// be wrapped up in a Ctor class.                                                                                                                                                                                                         
  /// The arguments that follow are the dimensions of the                                                                                                                                                                                    
  /// View objects.  For example, if the View has three dimensions,                                                                                                                                                                          
  /// the first three integer arguments will be nonzero, and you may                                                                                                                                                                         
  /// omit the integer arguments that follow.                                                                                                                                                                                                
  template <class... P>                                                                                                                                                                                                                      
  DualView(const Impl::ViewCtorProp<P...>& arg_prop,                                                                                                                                                                                         
           typename std::enable_if<!Impl::ViewCtorProp<P...>::has_pointer,                                                                                                                                                                   
                                   size_t>::type const n0 =                                                                                                                                                                                  
               KOKKOS_IMPL_CTOR_DEFAULT_ARG,                                                                                                                                                                                                 
           const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,                                                                                                                                                                                   
           const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,                                                                                                                                                                                   
           const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,                                                                                                                                                                                   
           const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,                                                                                                                                                                                   
           const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,                                                                                                                                                                                   
           const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,                                                                                                                                                                                   
           const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG);

  // Copy constructor (shallow copy)
  template <class SS, class LS, class DS, class MS>
  DualView(const DualView<SS, LS, DS, MS>& src);

  // Subview constructor
  template <class SD, class S1, class S2, class S3, class Arg0, class... Args>
  DualView(const DualView<SD, S1, S2, S3>& src, const Arg0& arg0, Args... args);


  // Create DualView from existing device and host View objects.
  //
  // This constructor assumes that the device and host View objects
  // are synchronized.  You, the caller, are responsible for making
  // sure this is the case before calling this constructor.  After
  // this constructor returns, you may use DualView's sync() and
  // modify() methods to ensure synchronization of the View objects.
  //
  // .  d_view_ Device View
  // .  h_view_ Host View (must have type t_host = t_dev::HostMirror)
  DualView(const t_dev& d_view_, const t_host& h_view_);

  // Methods for synchronizing, marking as modified, and getting Views.

  // Return a View on a specific device `Device`.
  //
  // Please don't be afraid of the if_c expression in the return
  // value's type.  That just tells the method what the return type
  // should be: t_dev if the \c Device template parameter matches
  // this DualView's device type, else t_host.
  //
  // For example, suppose you create a DualView on Cuda, like this:
  // 
  //  typedef Kokkos::DualView<float, Kokkos::LayoutRight, Kokkos::Cuda>
  //  dual_view_type; dual_view_type DV ("my dual view", 100); \endcode If you
  //  want to get the CUDA device View, do this: \code typename
  //  dual_view_type::t_dev cudaView = DV.view<Kokkos::Cuda> (); \endcode and if
  //  you want to get the host mirror of that View, do this: \code typedef
  //  typename Kokkos::HostSpace::execution_space host_device_type; typename
  //  dual_view_type::t_host hostView = DV.view<host_device_type> (); \endcode
  template <class Device>
  KOKKOS_INLINE_FUNCTION const typename Impl::if_c<
      std::is_same<typename t_dev::memory_space,
                   typename Device::memory_space>::value,
      t_dev, t_host>::type&
  view();

  template <class Device>
  static int get_device_side();

  //  Update data on device or host only if data in the other
  //  space has been marked as modified.
  //
  //  If `Device` is the same as this DualView's device type, then
  //  copy data from host to device.  Otherwise, copy data from device
  //  to host.  In either case, only copy if the source of the copy
  //  has been modified.
  //
  //  This is a one-way synchronization only.  If the target of the
  //  copy has been modified, this operation will discard those
  //  modifications.  It will also reset both device and host modified
  //  flags.
  //
  //   This method doesn't know on its own whether you modified
  //   the data in either View.  You must manually mark modified data
  //   as modified, by calling the modify() method with the
  //   appropriate template parameter.
  template <class Device>
  void sync(const typename Impl::enable_if<
                (std::is_same<typename traits::data_type,
                              typename traits::non_const_data_type>::value) ||
                    (std::is_same<Device, int>::value),
                int>::type& = 0) 

  template <class Device>
  void sync(const typename Impl::enable_if<
                (!std::is_same<typename traits::data_type,
                               typename traits::non_const_data_type>::value) ||
                    (std::is_same<Device, int>::value),
                int>::type& = 0); 

  template <class Device>
  bool need_sync() const;

  // Mark data as modified on the given device \c Device.
  //
  // If `Device` is the same as this DualView's device type, then
  // mark the device's data as modified.  Otherwise, mark the host's
  // data as modified.
  template <class Device>
  void modify() 

  inline void clear_sync_state();

  //Methods for reallocating or resizing the View objects.

  // Return allocation state of underlying views
  //
  // Returns true if both the host and device views points to a valid memory location.  
  // This function works for both managed and unmanaged views. With the unmanaged view, 
  // there is no guarantee that referenced address is valid, only that it is a non-null 
  // pointer. 
  constexpr bool is_allocated() const;
    
  // Reallocate both View objects.
  //
  // This discards any existing contents of the objects, and resets
  // their modified flags.  It does <i>not</i> copy the old contents
  // of either View into the new View objects.
  void realloc(const size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
               const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
               const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
               const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
               const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
               const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
               const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
               const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG);

  // Resize both views, copying old contents into new if necessary.
  //
  // This method only copies the old contents into the new View
  // objects for the device which was last marked as modified.
  void resize(const size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
              const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
              const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
              const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
              const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
              const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
              const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
              const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG);

  //  Methods for getting capacity, stride, or dimension(s).

  // The allocation size (same as Kokkos::View::span).
  KOKKOS_INLINE_FUNCTION constexpr size_t span() const;

  // Return true if the span is contiguous
  KOKKOS_INLINE_FUNCTION bool span_is_contiguous();

  // Get stride(s) for each dimension. Sets stride_[rank] to span().
  template <typename iType>
  void stride(iType* stride_) const;

  // return the extent for the requested rank
  template <typename iType>
  KOKKOS_INLINE_FUNCTION constexpr
      typename std::enable_if<std::is_integral<iType>::value, size_t>::type
      extent(const iType& r) const ;

  // return integral extent for the requested rank
  template <typename iType>
  KOKKOS_INLINE_FUNCTION constexpr
      typename std::enable_if<std::is_integral<iType>::value, int>::type
      extent_int(const iType& r) const;


};
```
