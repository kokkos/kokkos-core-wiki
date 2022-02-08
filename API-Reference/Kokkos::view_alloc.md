# Kokkos::view_alloc()

Header File: `Kokkos_View.hpp`

Usage:
```c++
  Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing, "ViewString");
  Kokkos::view_wrap(pointer_to_wrapping_memory);
```

Create View allocation parameter bundle from argument list. Valid argument list members are:
 * label as a "string" or std::string
 * memory space instance of the View::memory_space type
 * execution space instance compatible with the View::memory_space
 * Kokkos::WithoutInitializing to bypass initialization
 * Kokkos::AllowPadding to allow allocation to pad dimensions for memory alignment
 * a pointer to create an unmanaged View wrapping that pointer

## Synopsis

```cpp
template <class... Args>
Impl::ViewCtorProp<typename Impl::ViewCtorProp<void, Args>::type...>
view_alloc(Args const&... args);

template <class... Args>
KOKKOS_FUNCTION
Impl::ViewCtorProp<typename Impl::ViewCtorProp<void, Args>::type...>
view_wrap(Args const&... args);
```

## Description

* ```cpp
  template <class... Args>
  Impl::ViewCtorProp<typename Impl::ViewCtorProp<void, Args>::type...>
  view_alloc(Args const&... args);
  ```
  Create View allocation parameter bundle from argument list.

  Restrictions:
  * `args`: Cannot contain a pointer to memory.

* ```cpp
  template <class... Args>
  Impl::ViewCtorProp<typename Impl::ViewCtorProp<void, Args>::type...>
  view_alloc(Args const&... args);
  ```
  Create View allocation parameter bundle from argument list.

  Restrictions:
  * `args`: Cannot only be a pointer to memory.
