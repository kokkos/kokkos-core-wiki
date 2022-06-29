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
  template<class Integral> void scatter_to(T* mem, simd<Integral, typename T::abi_type> const& index) const;
};

template<class M, class T>
class where_expression : public const_where_expression<M, T> {
 public:
  template<class U> void operator=(U&& x);
  template<class U, class Flags> void copy_from(const U* mem, Flags);
  template <class Integral> void gather_from(T const* mem, simd<Integral, typename T::abi_type> const& index);
};
```

Either `M` is bool, in which case `T` can be anything,
or `M` is `simd<T2, Abi>::mask_type` and `T` is `simd<T2, Abi>`.

```c++
template<class U, class Flags> void copy_to(U* mem, Flags f) const;
```

If `M` is `bool`, copies as if `*mem = static_cast<U>(data)` if `mask` is `true`.
Otherwise, copies the selected elements as if `mem[i] = static_cast<U>(data[i])` for all selected indices `i`.

```c++
template<class Integral> void scatter_to(T* mem, simd<Integral, typename T::abi_type> const& index) const;
```

Copies the selected elements as if `mem[index[i]] = data[i]` for all selected indices `i`.

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

```c++
template <class Integral> void gather_from(T const* mem, simd<Integral, typename T::abi_type> const& index);
```

Copies the selected elements as if `data[i] = mem[index[i]]` for all selected indices `i`.

## Class template simd

```c++
template<class T, class Abi> class simd {
 public:
  using value_type = T;
  using reference = /* see below */;
  using mask_type = simd_mask<T, Abi>;
  using abi_type = Abi;
  static constexpr size_t size() noexcept;
  simd() noexcept = default;

  // simd constructors
  template<class U> simd(U&& value);
  template<class U> simd(const simd<U, simd_abi::fixed_size<size()>>&);
  template<class G> explicit simd(G&& gen);
  template<class U, class Flags> simd(const U* mem, Flags f);

  // simd copy functions
  template<class U, class Flags> copy_from(const U* mem, Flags f);
  template<class U, class Flags> copy_to(U* mem, Flags f);
  // simd subscript operators
  reference operator[](size_t);
  value_type operator[](size_t) const;

  // simd unary operators
  simd operator-() const;

  // simd binary operators
  friend simd operator+(const simd&, const simd&);
  friend simd operator-(const simd&, const simd&);
  friend simd operator*(const simd&, const simd&) noexcept;
  friend simd operator/(const simd&, const simd&) noexcept;
  friend simd operator<<(const simd&, const simd<int, Abi>&) noexcept;
  friend simd operator>>(const simd&, const simd<int, Abi>&) noexcept;
  friend simd operator<<(const simd&, int) noexcept;
  friend simd operator>>(const simd&, int) noexcept;

  // simd compound assignment
  friend simd& operator+=(simd&, const simd&) noexcept;
  friend simd& operator-=(simd&, const simd&) noexcept;
  friend simd& operator*=(simd&, const simd&) noexcept;
  friend simd& operator/=(simd&, const simd&) noexcept;
  // simd compare operators
  friend mask_type operator==(const simd&, const simd&) noexcept;
  friend mask_type operator!=(const simd&, const simd&) noexcept;
  friend mask_type operator>=(const simd&, const simd&) noexcept;
  friend mask_type operator<=(const simd&, const simd&) noexcept;
  friend mask_type operator>(const simd&, const simd&) noexcept;
  friend mask_type operator<(const simd&, const simd&) noexcept;
};
```
