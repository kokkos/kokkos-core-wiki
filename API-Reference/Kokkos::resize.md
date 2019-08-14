# `Kokkos::atomic_resize`

Header File: `Kokkos_Core.hpp`

Usage:
  ```c++
  resize(view,n0,n1,n2,n3);
  resize(view,layout);
  ```

Reallocates a view to have the new dimensions. Can grow or shrink, and will not preserve content.

## Synopsis

```c++
template <class T, class... P>
void resize(View<T, P...>& v,
        const size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG);

template <class T, class... P>
void resize(Kokkos::View<T, P...>& v,
             const typename Kokkos::View<T, P...>::array_layout& layout); 
```

## Description


* ```c++
  template <class T, class... P>
  void resize(View<T, P...>& v,
        const size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG);
  ```
  Resizes `v` to have the new dimensions while preserving the contents for the common subview of the old and new view.
  * `v`: existing view, can be a default constructed one. 
  * `n[X]`: new length for extent X.

  Restrictions:
  * `View<T, P...>::array_layout` is either `LayoutLeft` or `LayoutRight`.

* ```c++
  template <class T, class... P>
  void resize(Kokkos::View<T, P...>& v,
             const typename Kokkos::View<T, P...>::array_layout& layout); 
  ```
  Resizes `v` to have the new dimensions while preserving the contents for the common subview of the old and new view.
  * `v`: existing view, can be a default constructed one. 
  * `layout`: a layout instance containing the new dimensions.



