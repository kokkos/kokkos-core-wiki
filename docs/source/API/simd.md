# SIMD Types

Header File: `Kokkos_SIMD.hpp`

This library is based on ISO C++ document N4808 titled "Working Draft, C++ Extensions for Parallelism Version 2".

## SIMD ABI tags

```c++
namespace Kokkos::Experimental::simd_abi {
  using scalar = ...
  using host_native = ...
  using device_native = ...
}
```

These ABI tag types are used to select an implementation of SIMD types, usually choosing a particular
instruction set architecture.
In Kokkos, we provide both `host_native` and `device_native`, which are the most appropriate SIMD
ABI type to use for the host and device hardware, respectively.

## Where expression class templates

```c++
template<class M, class T>
class const_where_expression {
  const M mask; // exposition only
  T& data; // exposition only
 public:
  template<class U, class Flags> void copy_to(U* mem, Flags f) const;
};

template<class M, class T>
class where_expression : public const_where_expression<M, T> {
 public:
  template<class U> void operator=(U&& x);
  template<class U, class Flags> void copy_from(const U* mem, Flags);
};
```

Either `M` is bool, in which case `T` can be anything,
or `M` is `simd<T2, Abi>::mask_type` and `T` is `simd<T2, Abi>`.

```c++
template<class U, class Flags> void copy_to(U* mem, Flags f) const;
```

If `M` is `bool`, copies as if `*mem = static_cast<U>(data)` if `mask` is `true`.
Otherwise, copies the selected elements as if `mem[i] = static_cast<U>(data[i])` for all selected indices `i`.

```
template<class U> void operator=(U&& x);
```

If `M` is `bool`, replaces `data` with `static_cast<T>(std::forward<U>(x))` if `mask` is `true`.
Otherwise, replaces `data[i]` with `static_cast<T>(std::forward<U>(x))[i]` for all selected indices `i`.

```c++
template<class U, class Flags> void copy_from(const U* mem, Flags);
```

If `M` is `bool`, copies as if `data = static_cast<T>(*mem)` if `mask` is `true`.
Otherwise, copies the selected elements as if `data[i] = static_cast<T>(mem[i])` for all selected indices `i`.
