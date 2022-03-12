# `Kokkos::create_mirror[_view]`

Header File: `Kokkos_Core.hpp`

A common desired use case is to have a memory allocation in GPU memory and an identical memory allocation in CPU memory, such that copying from one to another is straightforward. To satisfy this use case and others, Kokkos has facilities for dealing with "mirrors" of View. A "mirror" of a View type `A` is loosely defined a View type `B` such that Views of type `B` are accessible from the CPU and `deep_copy` between Views of type `A` and `B` are direct. The most common functions for dealing with mirrors are `create_mirror`, `create_mirror_view` and `create_mirror_view_and_copy`.

Usage:
```cpp
auto host_mirror = create_mirror(a_view);
auto host_mirror_view = create_mirror_view(a_view);

auto host_mirror_space = create_mirror(ExecSpace(),a_view);
auto host_mirror_view_space = create_mirror_view(ExecSpace(),a_view);
```

## Synopsis

```cpp

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

template <class Space, class ViewType>
ImplMirrorType create_mirror_view_and_copy(Space const& space, ViewType const&);
```


## Description

* ```cpp
  template <class ViewType>
  typename ViewType::HostMirror create_mirror(ViewType const& src);
  ```
  Creates a new host accessible `View` with the same layout and padding as `src`.
  * `src`: a `Kokkos::View`.

* ```cpp
  template <class ViewType>
  typename ViewType::HostMirror create_mirror(decltype(Kokkos::ViewAllocateWithoutInitializing()),
                                              ViewType const& src);
  ```
  Creates a new host accessible `View` with the same layout and padding as `src`. The new view will have uninitialized data.
  * `src`: a `Kokkos::View`.

* ```cpp
  template <class Space, class ViewType>
  ImplMirrorType create_mirror(Space const& space, ViewType const&);
  ```
  Creates a new `View` with the same layout and padding as `src` but with a device type of `Space::device_type`.
  * `src`: a `Kokkos::View`.
  * `Space`: a class meeting the requirements of `ExecutionSpaceConcept` or `MemorySpaceConcept`
  * `ImplMirrorType`: an implementation defined specialization of `Kokkos::View`.

* ```cpp
  template <class Space, class ViewType>
  ImplMirrorType create_mirror(decltype(Kokkos::ViewAllocateWithoutInitializing()),
                               Space const& space, ViewType const&);
  ```
  Creates a new `View` with the same layout and padding as `src` but with a device type of `Space::device_type`. The new view will have uninitialized data.
  * `src`: a `Kokkos::View`.
  * `Space`: a class meeting the requirements of `ExecutionSpaceConcept` or `MemorySpaceConcept`
  * `ImplMirrorType`: an implementation defined specialization of `Kokkos::View`.

* ```cpp
  template <class ViewType>
  typename ViewType::HostMirror create_mirror_view(ViewType const& src);
  ```
  If `src` is not host accessible (i.e. if `SpaceAccessibility<HostSpace,ViewType::memory_space>::accessible` is `false`)
  it creates a new host accessible `View` with the same layout and padding as `src`. Otherwise returns `src`.
  * `src`: a `Kokkos::View`.

* ```cpp
  template <class ViewType>
  typename ViewType::HostMirror create_mirror_view(decltype(Kokkos::ViewAllocateWithoutInitializing()),
                                                   ViewType const& src);
  ```
  If `src` is not host accessible (i.e. if `SpaceAccessibility<HostSpace,ViewType::memory_space>::accessible` is `false`)
  it creates a new host accessible `View` with the same layout and padding as `src`. The new view will have uninitialized data. Otherwise returns `src`.
  * `src`: a `Kokkos::View`.

* ```cpp
  template <class Space, class ViewType>
  ImplMirrorType create_mirror_view(Space const& space, ViewType const&);
  ```
  If `std::is_same<typename Space::memory_space, typename ViewType::memory_space>::value` is `false`,
  creates a new `View` with the same layout and padding as `src` but with a device type of `Space::device_type`.
  Otherwise returns `src`.
  * `src`: a `Kokkos::View`.
  * `Space`: a class meeting the requirements of `ExecutionSpaceConcept` or `MemorySpaceConcept`
  * `ImplMirrorType`: an implementation defined specialization of `Kokkos::View`.

* ```cpp
  template <class Space, class ViewType>
  ImplMirrorType create_mirror_view(decltype(Kokkos::ViewAllocateWithoutInitializing()),
                                    Space const& space, ViewType const&);
  ```
  If `std::is_same<typename Space::memory_space, typename ViewType::memory_space>::value` is `false`,
  creates a new `View` with the same layout and padding as `src` but with a device type of `Space::device_type`. The new view will have uninitialized data.
  Otherwise returns `src`.
  * `src`: a `Kokkos::View`.
  * `Space`: a class meeting the requirements of `ExecutionSpaceConcept` or `MemorySpaceConcept`
  * `ImplMirrorType`: an implementation defined specialization of `Kokkos::View`.

* ```cpp
  template <class Space, class ViewType>
  ImplMirrorType create_mirror_view_and_copy(Space const& space, ViewType const&);
  ```
  If `std::is_same<typename Space::memory_space, typename ViewType::memory_space>::value` is `false`,
  creates a new `Kokkos::View` with the same layout and padding as `src` but with a device type of `Space::device_type` and
  conducts a `deep_copy` from `src` to the new view if one was created. Otherwise returns `src`.
  * `src`: a `Kokkos::View`.
  * `Space`: a class meeting the requirements of `ExecutionSpaceConcept` or `MemorySpaceConcept`
  * `ImplMirrorType`: an implementation defined specialization of `Kokkos::View`.
