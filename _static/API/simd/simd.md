# `Experimental::simd`

Header File: `Kokkos_SIMD.hpp`

Usage: 

`Kokkos::Experimental::simd` is an abstraction over platform-specific vector datatypes and calls platform-specific vector intrinsics.
It is based on the `simd` type proposed for ISO C++ in [this document](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2019/n4808.pdf)

## Interface

```c++
namespace Experimental {
template <class T, class Abi>
class simd;
}
```

### Template Parameters

The first template parameter `T` should be a C++ fundamental type for which the current platform supports vector intrinsics. Kokkos supports the following types for `T`:
 - `double`
 - `std::int32_t`
 - `std::int64_t`
 - `std::uint32_t`
 - `std::uint64_t`

The second template parameter `Abi` is one of the pre-defined ABI types in the namespace `Kokkos::Experimental::simd_abi`. This type determines the size of the vector and what architecture-specific intrinsics will be used. The following types are always available in that namespace:
 - `scalar`: a fallback ABI which always has vector size of 1 and uses no special intrinsics.
 - `native`: the "best" ABI for the architecture for which Kokkos was compiled.

### Typedefs

 *  `value_type`: Equal to `T`
 *  `reference`: This type should be convertible to `value_type` and `value_type` should be assignable to `reference`. It may be a plain reference or it may be an implementation-defined type that calls vector intrinsics to extract or fill in one vector lane.
 *  `mask_type`: Equal to `simd_mask<T, Abi>`
 *  `abi_type`: Equal to `Abi`

### Width

 * `static constexpr std::size_t size()`: `simd<T, Abi>::size()` is a compile-time constant of the width of the vector, i.e. the number of values of type `T` in the vector.

### Constructors

  * `simd()`: Default Constructor. The vector values are not initialized by this constructor.
  * `template <class U> simd(U&&)`: Single-value constructor. The argument will be converted to `value_type` and all the vector values will be set equal to that value.
  * `template <class G> simd(G&& gen)`: Generator constructor. The generator `gen` should be a callable type (e.g. functor) that can accept `std::integral_constant<std::size_t, i>()` as an argument and return something convertible to `value_type`. Vector lane `i` will be initialized to the value of `gen(std::integral_constant<std::size_t, i>())`.
  * `template <class U, class Flags> simd(const U* mem, Flags)`: loads the vector values from memory based on the guidance represented by the `Flags`. The only supported type for `Flags` is `Kokkos::Experimental::element_aligned_tag`.

### Load/Store Methods

  * `template <class U, class Flags> void copy_from(const U* mem, Flags flags)`: Loads the full vector of contiguous values starting at the address `mem`. The only valid type for `Flags` is `Kokkos::Experimental::element_aligned_tag`.
  * `template <class U, class Flags> void copy_to(U* mem, Flags flags)`: Stores the full vector of contiguous values starting at the address `mem`. The only valid type for `Flags` is `Kokkos::Experimental::element_aligned_tag`.

### Value Access Methods
  * `reference operator[](std::size_t)`: returns a reference to vector value `i` that can be modified.
  * `value_type operator[](std::size_t) const`: returns the vector value `i`.

### Arithmetic Operators
  * `simd simd::operator-() const`
  * `simd operator+(const simd& lhs, const simd& rhs)`
  * `simd operator-(const simd& lhs, const simd& rhs)`
  * `simd operator*(const simd& lhs, const simd& rhs)`
  * `simd operator/(const simd& lhs, const simd& rhs)`
  * `simd operator>>(const simd& lhs, const simd& rhs)`
  * `simd operator<<(const simd& lhs, const simd& rhs)`
  * `simd operator+=(simd& lhs, const simd& rhs)`
  * `simd operator-=(simd& lhs, const simd& rhs)`
  * `simd operator*=(simd& lhs, const simd& rhs)`
  * `simd operator/=(simd& lhs, const simd& rhs)`

### Comparison Operators
  * `mask_type operator==(const simd& lhs, const simd& rhs)`
  * `mask_type operator!=(const simd& lhs, const simd& rhs)`
  * `mask_type operator>=(const simd& lhs, const simd& rhs)`
  * `mask_type operator<=(const simd& lhs, const simd& rhs)`
  * `mask_type operator>(const simd& lhs, const simd& rhs)`
  * `mask_type operator<(const simd& lhs, const simd& rhs)`

### Min/Max Functions
These functions are defined for all supported `value_type`s.
  * `simd Kokkos::min(const simd& lhs, const simd& rhs)`
  * `simd Kokkos::max(const simd& lhs, const simd& rhs)`

### `<cmath>` Functions
These functions are only defined for `value_type=double`.
  * `simd Kokkos::abs(const simd& lhs)`
  * `simd Kokkos::exp(const simd& lhs)`
  * `simd Kokkos::exp2(const simd& lhs)`
  * `simd Kokkos::log(const simd& lhs)`
  * `simd Kokkos::log10(const simd& lhs)`
  * `simd Kokkos::sqrt(const simd& lhs)`
  * `simd Kokkos::cbrt(const simd& lhs)`
  * `simd Kokkos::sin(const simd& lhs)`
  * `simd Kokkos::cos(const simd& lhs)`
  * `simd Kokkos::tan(const simd& lhs)`
  * `simd Kokkos::asin(const simd& lhs)`
  * `simd Kokkos::acos(const simd& lhs)`
  * `simd Kokkos::atan(const simd& lhs)`
  * `simd Kokkos::sinh(const simd& lhs)`
  * `simd Kokkos::cosh(const simd& lhs)`
  * `simd Kokkos::tanh(const simd& lhs)`
  * `simd Kokkos::asinh(const simd& lhs)`
  * `simd Kokkos::acosh(const simd& lhs)`
  * `simd Kokkos::atanh(const simd& lhs)`
  * `simd Kokkos::erf(const simd& lhs)`
  * `simd Kokkos::erfc(const simd& lhs)`
  * `simd Kokkos::tgamma(const simd& lhs)`
  * `simd Kokkos::lgamma(const simd& lhs)`
  * `simd Kokkos::pow(const simd& lhs, const simd& rhs)`
  * `simd Kokkos::hypot(const simd& x, const simd& y)`
  * `simd Kokkos::hypot(const simd& x, const simd& y, const simd& z)`
  * `simd Kokkos::atan2(const simd& x, const simd& y)`
  * `simd Kokkos::copysign(const simd& mag, const simd& sgn)`
  * `simd Kokkos::fma(const simd& x, const simd& y, const simd& z)`

### Global Typedefs
  * `template <class T> Kokkos::Experimental::native_simd`: Alias for `Kokkos::Experimental::simd<T, Kokkos::Experimental::simd_abi::native<T>>`.

## Examples

```c++
#include <Kokkos_SIMD.hpp>
#include <cstdio>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc,argv);
  {
  using simd_type = Kokkos::Experimental::native_simd<double>;
  simd_type a([] (std::size_t i) { return 0.1 * i; });
  simd_type b(2.0);
  simd_type c = Kokkos::sqrt(a * a + b * b);
  for (std::size_t i = 0; i < simd_type::size(); ++i) {
    printf("[%zu] = %g\n", i, c[i]);
  }
  }
  Kokkos::finalize();
}
```
