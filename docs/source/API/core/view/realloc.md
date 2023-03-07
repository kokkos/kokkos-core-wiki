# `realloc`

Header File: `Kokkos_Core.hpp`

Usage:
  ```c++
  realloc(view,n0,n1,n2,n3);
  realloc(view,layout);
  ```

Reallocates a view to have the new dimensions. Can grow or shrink, and will not preserve content.
May not modify the view, if sizes already match.

## Synopsis

```c++
template <class T, class... P>
void realloc(View<T, P...>& v,
        const size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
        const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG);

template <class I, class T, class... P>
void realloc(const I& arg_prop, Kokkos::View<T, P...>& v,
       const size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG);

template <class... ViewCtorArgs, class T, class... P>
void realloc(const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop,
       Kokkos::View<T, P...>& v,
       const size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
       const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG);

template <class T, class... P>
void realloc(Kokkos::View<T, P...>& v,
       const typename Kokkos::View<T, P...>::array_layout& layout);

template <class I, class T, class... P>
void realloc(const I& arg_prop, Kokkos::View<T, P...>& v,
       const typename Kokkos::View<T, P...>::array_layout& layout);

template <class... ViewCtorArgs, class T, class... P>
void realloc(const Impl::ViewCtorProp<ViewCtorArgs...>& arg_prop,
             Kokkos::View<T, P...>& v,
             const typename Kokkos::View<T, P...>::array_layout& layout);
```

## Description


* ```c++
  template <class T, class... P>
  void realloc(View<T, P...>& v,
         const size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
         const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
         const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
         const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
         const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
         const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
         const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
         const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG);
  ```
  Resizes `v` to have the new dimensions without preserving its contents.
  May not modify `v` if the dimensions already match.
  * `v`: existing view, can be a default constructed one.
  * `n[X]`: new length for extent X.

  Restrictions:
  * `View<T, P...>::array_layout` is either `LayoutLeft` or `LayoutRight`.

* ```c++
  template <class I, class T, class... P>
  void realloc(const I& arg_prop, Kokkos::View<T, P...>& v,
         const size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
         const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
         const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
         const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
         const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
         const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
         const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
         const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG);
  ```
  Resizes `v` to have the new dimensions without preserving its contents. The new `Kokkos::View` is constructed using the View constructor property `arg_prop`, e.g., Kokkos::WithoutInitializing.
  May not modify `v` if the dimensions already match.
  * `v`: existing view, can be a default constructed one.
  * `n[X]`: new length for extent X.
  * `arg_prop`: View constructor property, e.g., `Kokkos::WithoutInitializing`.

  Restrictions:
  * `View<T, P...>::array_layout` is either `LayoutLeft` or `LayoutRight`.

* ```c++
  template <class... ViewCtorArgs, class T, class... P>
  void realloc(const I& arg_prop, Kokkos::View<T, P...>& v,
         Kokkos::View<T, P...>& v,
         const size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
         const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
         const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
         const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
         const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
         const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
         const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
         const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG);
  ```
  Resizes `v` to have the new dimensions without preserving its contents. The new `Kokkos::View` is constructed using the View constructor properties `arg_prop`, e.g., `Kokkos::view_alloc(Kokkos::WithoutInitializing)`.
  May not modify `v` if the dimensions already match.
  * `v`: existing view, can be a default constructed one.
  * `n[X]`: new length for extent X.
  * `arg_prop`: View constructor properties, e.g., `Kokkos::view_alloc(Kokkos::WithoutInitializing)`.

  Restrictions:
  * `View<T, P...>::array_layout` is either `LayoutLeft` or `LayoutRight`.
  * `arg_prop` must not include a pointer to memory, a label, or a memory space.

* ```c++
  template <class T, class... P>
  void realloc(Kokkos::View<T, P...>& v,
         const typename Kokkos::View<T, P...>::array_layout& layout);
  ```
  Resizes `v` to have the new dimensions without preserving its contents.
  May not modify `v` if the dimensions already match.
  * `v`: existing view, can be a default constructed one.
  * `layout`: a layout instance containing the new dimensions.

* ```c++
  template <class I, class T, class... P>
  void realloc(const I& arg_prop, Kokkos::View<T, P...>& v,
         const typename Kokkos::View<T, P...>::array_layout& layout);
  ```
  Resizes `v` to have the new dimensions without preserving its contents. The new `Kokkos::View` is constructed using the View constructor property `arg_prop`, e.g., Kokkos::WithoutInitializing.
  May not modify `v` if the dimensions already match.
  * `v`: existing view, can be a default constructed one.
  * `layout`: a layout instance containing the new dimensions.
  * `arg_prop`: View constructor property, e.g., `Kokkos::WithoutInitializing`.

* ```c++
  template <class... ViewCtorArgs, class T, class... P>
  void realloc(const I& arg_prop, Kokkos::View<T, P...>& v,
         Kokkos::View<T, P...>& v,
         const typename Kokkos::View<T, P...>::array_layout& layout);
  ```
  Resizes `v` to have the new dimensions without preserving its contents. The new `Kokkos::View` is constructed using the View constructor properties `arg_prop`, e.g., `Kokkos::view_alloc(Kokkos::WithoutInitializing)`.
  May not modify `v` if the dimensions already match.
  * `v`: existing view, can be a default constructed one.
  * `layout`: a layout instance containing the new dimensions.
  * `arg_prop`: View constructor properties, e.g., `Kokkos::view_alloc(Kokkos::WithoutInitializing)`.

  Restrictions:
  * `arg_prop` must not include a pointer to memory, a label, or a memory space.

## Possibly Unexpected Behavior Warning

`realloc` will only modify the specific `View` instance passed to it.
Any other `View` which aliases the same allocation will be unmodified.
Consequently, if the `use_count()` of the `View` is larger than 1, the
old allocation will not be deleted.
Note that if the size arguments already match the extents of the `View`
argument, that `realloc` may not create a new `View`.

## Example:
  * ```c++
    Kokkos::realloc(v, 2, 3);
    ```
    Reallocate a `Kokkos::View` with dynamic rank 2 to have dynamic extent 2 and 3 respectively.
  * ```c++
    Kokkos::realloc(Kokkos::WithoutInitializing, v, 2, 3); 
    ```
    Reallocate a `Kokkos::View` with dynamic rank 2 to have dynamic extent 2 and 3 respectively. After this call, the View is uninitialized.
