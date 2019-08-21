# DualView

Container to manage mirroring a Kokkos::View that references device memory with a Kokkos::View that references host memory.  The class provides capabilities to manage data which exists in two different memory spaces at the same time.  It supports views with the same layout on two memory spaces as well as modified flags for both allocations.  Users are responsible for updating the modified flags manually if they change the data in either memory space, by calling the sync() method, which is templated on the device with the modified data.  Users may also synchronize data by calling the modify() function, which is templated on the device that requires synchronization (i.e., the target of the one-way copy operation).
 
The DualView class also provides convenience methods such as realloc, resize and capacity which call the appropriate methods of the underlying Kokkos::View objects.
 
The four template arguments are the same as those of Kokkos::View.
 
 - DataType, The type of the entries stored in the container.
 
 - Layout, The array's layout in memory.
 
 - Device, The Kokkos Device type.  If its memory space is not the same as the host's memory space, then DualView will contain two separate Views: one in device memory, and one in host memory.  Otherwise, DualView will only store one View.
 
 - MemoryTraits (optional) The user's intended memory access behavior.  Please see the documentation of Kokkos::View for examples.  The default suffices for most users.

Usage:

`   using view_type = Kokkos::DualView<Scalar**, Kokkos::LayoutLeft, Device>
    view_type a("A", n, m);

    Kokkos::deep_copy(a.d_view, 1);
    a.template modify<typename view_type::execution_space>();
    a.template sync<typename view_type::host_mirror_space>();

    Kokkos::deep_copy(a.h_view, 2);
    a.template modify<typename ViewType::host_mirror_space>();
    a.template sync<typename ViewType::execution_space>();
`
