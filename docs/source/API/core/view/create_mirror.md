# `create_mirror[_view]`

Header File: `Kokkos_Core.hpp`

A common desired use case is to have a memory allocation in GPU memory and an identical memory allocation in CPU memory, such that copying from one to another is straightforward. To satisfy this use case and others, Kokkos has facilities for dealing with "mirrors" of View. A "mirror" of a View type `A` is loosely defined a View type `B` such that Views of type `B` are accessible from the CPU and [`deep_copy`](deep_copy) between Views of type `A` and `B` are direct. The most common functions for dealing with mirrors are `create_mirror`, `create_mirror_view` and `create_mirror_view_and_copy`.

Usage:
```c++
auto host_mirror = create_mirror(a_view);
auto host_mirror_view = create_mirror_view(a_view);

auto host_mirror_space = create_mirror(ExecSpace(),a_view);
auto host_mirror_view_space = create_mirror_view(ExecSpace(),a_view);
```

## Synopsis

```c++
template <class ViewType>
typename ViewType::HostMirror create_mirror(ViewType const&);

template <class ViewType>
typename ViewType::HostMirror create_mirror(decltype(Kokkos::ViewAllocateWithoutInitializing()),
                                            ViewType const&);

template <class Space, class ViewType>
ImplMirrorType create_mirror(Space const& space, ViewType const&);

template <class Space, class ViewType>
ImplMirrorType create_mirror(decltype(Kokkos::ViewAllocateWithoutInitializing()),
                             Space const& space, ViewType const&);

template <class ViewType, class... ViewCtorArgs>
auto create_mirror(Impl::ViewCtorProp<ViewCtorArgs...> const& arg_prop,
                   ViewType const& v);

template <class ViewType>
typename ViewType::HostMirror create_mirror_view(ViewType const&);

template <class ViewType>
typename ViewType::HostMirror create_mirror_view(decltype(Kokkos::ViewAllocateWithoutInitializing()),
                                                 ViewType const&);

template <class Space, class ViewType>
ImplMirrorType create_mirror_view(Space const& space, ViewType const&);

template <class Space, class ViewType>
ImplMirrorType create_mirror_view(decltype(Kokkos::ViewAllocateWithoutInitializing()),
                                  Space const& space, ViewType const&);

template <class ViewType, class... ViewCtorArgs>
auto create_mirror_view(Impl::ViewCtorProp<ViewCtorArgs...> const& arg_prop,
                        ViewType const& v);

template <class Space, class ViewType>
ImplMirrorType create_mirror_view_and_copy(Space const& space, ViewType const&);

template <class ViewType, class... ViewCtorArgs>
auto create_mirror_view_and_copy(Impl::ViewCtorProp<ViewCtorArgs...> const& arg_prop,
                                 ViewType const& v);
```


## Description

* ```c++
  template <class ViewType>
  typename ViewType::HostMirror create_mirror(ViewType const& src);
  ```
  Creates a new host accessible [`View`](view) with the same layout and padding as `src`.
  * `src`: a `Kokkos::View`.

* ```c++
  template <class ViewType>
  typename ViewType::HostMirror create_mirror(decltype(Kokkos::ViewAllocateWithoutInitializing()),
                                              ViewType const& src);
  ```
  Creates a new host accessible [`View`](view) with the same layout and padding as `src`. The new view will have uninitialized data.
  * `src`: a `Kokkos::View`.

* ```c++
  template <class Space, class ViewType>
  ImplMirrorType create_mirror(Space const& space, ViewType const&);
  ```
  Creates a new [`View`](view) with the same layout and padding as `src` but with a device type of `Space::device_type`.
  * `src`: a `Kokkos::View`.
  * `Space`: a class meeting the requirements of [`ExecutionSpaceConcept`](ExecutionSpaceConcept) or [`MemorySpaceConcept`](MemorySpaceConcept)
  * `ImplMirrorType`: an implementation defined specialization of `Kokkos::View`.

* ```c++
  template <class Space, class ViewType>
  ImplMirrorType create_mirror(decltype(Kokkos::ViewAllocateWithoutInitializing()),
                               Space const& space, ViewType const&);
  ```
  Creates a new [`View`](view) with the same layout and padding as `src` but with a device type of `Space::device_type`. The new view will have uninitialized data.
  * `src`: a `Kokkos::View`.
  * `Space`: a class meeting the requirements of [`ExecutionSpaceConcept`](ExecutionSpaceConcept) or [`MemorySpaceConcept`](MemorySpaceConcept)
  * `ImplMirrorType`: an implementation defined specialization of `Kokkos::View`.

* ```c++
  template <class ViewType, class... ViewCtorArgs>
  auto create_mirror(Impl::ViewCtorProp<ViewCtorArgs...> const& arg_prop,
                     ViewType const& v);
  ```
  Creates a new [`View`](view) with the same layout and padding as `src` using the [`View`](view) constructor properties `arg_prop`, e.g., `Kokkos::view_alloc(Kokkos::WithoutInitializing)`. If `arg_prop` contains a memory space, a [`View`](view) in that space is created. Otherwise, a [`View`](view) in host-accessible memory is returned.
  * `src`: a `Kokkos::View`.
  * `arg_prop`: [`View`](view) constructor properties, e.g., `Kokkos::view_alloc(Kokkos::WithoutInitializing)`.

 Restrictions:
  * `arg_prop` must not include a pointer to memory, or a label, or allow padding.

* ```c++
  template <class ViewType>
  typename ViewType::HostMirror create_mirror_view(ViewType const& src);
  ```
  If `src` is not host accessible (i.e. if `SpaceAccessibility<HostSpace,ViewType::memory_space>::accessible` is `false`)
  it creates a new host accessible [`View`](view) with the same layout and padding as `src`. Otherwise returns `src`.
  * `src`: a `Kokkos::View`.

* ```c++
  template <class ViewType>
  typename ViewType::HostMirror create_mirror_view(decltype(Kokkos::ViewAllocateWithoutInitializing()),
                                                   ViewType const& src);
  ```
  If `src` is not host accessible (i.e. if `SpaceAccessibility<HostSpace,ViewType::memory_space>::accessible` is `false`)
  it creates a new host accessible [`View`](view) with the same layout and padding as `src`. The new view will have uninitialized data. Otherwise returns `src`.
  * `src`: a `Kokkos::View`.

* ```c++
  template <class Space, class ViewType>
  ImplMirrorType create_mirror_view(Space const& space, ViewType const&);
  ```
  If `std::is_same<typename Space::memory_space, typename ViewType::memory_space>::value` is `false`,
  creates a new [`View`](view) with the same layout and padding as `src` but with a device type of `Space::device_type`.
  Otherwise returns `src`.
  * `src`: a `Kokkos::View`.
  * `Space`: a class meeting the requirements of [`ExecutionSpaceConcept`](ExecutionSpaceConcept) or [`MemorySpaceConcept`](MemorySpaceConcept)
  * `ImplMirrorType`: an implementation defined specialization of `Kokkos::View`.

* ```c++
  template <class Space, class ViewType>
  ImplMirrorType create_mirror_view(decltype(Kokkos::ViewAllocateWithoutInitializing()),
                                    Space const& space, ViewType const& src);
  ```
  If `std::is_same<typename Space::memory_space, typename ViewType::memory_space>::value` is `false`,
  creates a new [`View`](view) with the same layout and padding as `src` but with a device type of `Space::device_type`. The new view will have uninitialized data.
  Otherwise returns `src`.
  * `src`: a `Kokkos::View`.
  * `Space`: a class meeting the requirements of [`ExecutionSpaceConcept`](ExecutionSpaceConcept) or [`MemorySpaceConcept`](MemorySpaceConcept)
  * `ImplMirrorType`: an implementation defined specialization of `Kokkos::View`.

* ```c++
  template <class ViewType, class... ViewCtorArgs>
  auto create_mirror_view(Impl::ViewCtorProp<ViewCtorArgs...> const& arg_prop,
                          ViewType const& src);
  ```
  If the [`View`](view) constructor arguments `arg_prop` include a memory space and the memory space doesn't match the memory space of `src`, creates a new [`View`](view) in the specified memory_space.
  If the `arg_prop` don't include a memory space and the memory space of `src` is not host-accessible, creates a new host-accessible [`View`](view).
  Otherwise, `src` is returned.
  If a new [`View`](view) is created, the implicitly called constructor respects `arg_prop` and uses the same layout and padding as `src`.
  * `src`: a `Kokkos::View`.
  * `arg_prop`: [`View`](view) constructor properties, e.g., `Kokkos::view_alloc(Kokkos::WithoutInitializing)`.

 Restrictions:
  * `arg_prop` must not include a pointer to memory, or a label, or allow padding.

* ```c++
  template <class Space, class ViewType>
  ImplMirrorType create_mirror_view_and_copy(Space const& space, ViewType const&);
  ```
  If `std::is_same<typename Space::memory_space, typename ViewType::memory_space>::value` is `false`,
  creates a new `Kokkos::View` with the same layout and padding as `src` but with a device type of `Space::device_type` and
  conducts a `deep_copy` from `src` to the new view if one was created. Otherwise returns `src`.
  * `src`: a `Kokkos::View`.
  * `Space`: a class meeting the requirements of [`ExecutionSpaceConcept`](ExecutionSpaceConcept) or [`MemorySpaceConcept`](MemorySpaceConcept)
  * `ImplMirrorType`: an implementation defined specialization of `Kokkos::View`.

* ```c++
  template <class ViewType, class... ViewCtorArgs>
  ImplMirrorType create_mirror_view_and_copy(Impl::ViewCtorProp<ViewCtorArgs...> const& arg_prop,
                                             ViewType const& src);
  ```
  If the  memory space included in the [`View`](view) constructor arguments `arg_prop` matches the memory space of `src`, creates a new [`View`](view) in the specified memory space using `arg_prop` and the same layout andf padding as `src`. Additionally, a `deep_copy` from `src` to the new view is executed (using the execution space contained in `arg_prop` if provided).
Otherwise returns `src`.
  * `src`: a `Kokkos::View`.
  * `arg_prop`: [`View`](view) constructor properties, e.g., `Kokkos::view_alloc(Kokkos::HostSpace{}, Kokkos::WithoutInitializing)`.

 Restrictions:
  * `arg_prop` must not include a pointer to memory, or a label, or allow padding.
  * `arg_prop` must include a memory space.
