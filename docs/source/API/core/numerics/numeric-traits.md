# Numeric traits

Defined in
header [`<Kokkos_NumericTraits.hpp>`](https://github.com/kokkos/kokkos/blob/3.5.00/core/src/Kokkos_NumericTraits.hpp)
which is included from `<Kokkos_Core.hpp>`

Provides a replacement for [`numeric_limits` from the standard library header
`<limits>`](https://en.cppreference.com/w/cpp/types/numeric_limits).
Implements a new facility that is being to the C++23 standard library and that
breaks the monolithic `numeric_limits` class template apart into individual
trait templates.  For details, please refer to
[P1841](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p1841r2.pdf).

Numeric traits are defined in the `Kokkos::Experimental` namespace since Kokkos 3.5

See below the list of available traits.

---

_`trait`_\* denotes traits that were added in Kokkos 3.6

**Numeric distinguished value traits**
`infinity`
`finite_min`
`finite_max`
`epsilon`
`round_error`
`norm_min`
`denorm_min`*
`reciprocal_overflow_threshold`*
`quiet_NaN`*
`signaling_NaN`*

**Numeric characteristics traits**
`digits`
`digits10`
`max_digits10`
`radix`
`min_exponent`
`min_exponent10`
`max_exponent`
`max_exponent10`

---

| Standard library                                      | Kokkos with C++17                            |
|-------------------------------------------------------|----------------------------------------------|
| `std::numeric_limits<Integral>::min()`                | `finite_min_v<Integral>`                     |
| `std::numeric_limits<FloatingPoint>::min()`           | `norm_min_v<FloatingPoint>`                  |
| `std::numeric_limits<Arithmetic>::lowest()`           | `finite_min_v<Arithmetic>`                   |
| `std::numeric_limits<Arithmetic>::max()`              | `finite_max_v<Arithmetic>`                   |
| `std::numeric_limits<FloatingPoint>::epsilon()`       | `epsilon_v<FloatingPoint>`                   |
| `std::numeric_limits<FloatingPoint>::round_error()`   | `round_error_v<FloatingPoint>`               |
| `std::numeric_limits<FloatingPoint>::infinity()`      | `infinity_v<FloatingPoint>`                  |
| `std::numeric_limits<FloatingPoint>::quiet_NaN()`     | `quiet_NaN_v<FloatingPoint>` (since 3.6)     |
| `std::numeric_limits<FloatingPoint>::signaling_NaN()` | `signaling_NaN_v<FloatingPoint>` (since 3.6) |
| `std::numeric_limits<FloatingPoint>::denorm_min()`    | `denorm_min_v<FloatingPoint>` (since 3.6)    |
| `std::numeric_limits<Arithmetic>::digits`             | `digits_v<Arithmetic>`                       |
| `std::numeric_limits<Arithmetic>::digits10`           | `digits10_v<Arithmetic>`                     |
| `std::numeric_limits<FloatingPoint>::max_digits10`    | `max_digits10_v<FloatingPoint>`              |
| `std::numeric_limits<Arithmetic>::radix`              | `radix_v<Arithmetic>`                        |
| `std::numeric_limits<FloatingPoint>::min_exponent`    | `min_exponent_v<FloatingPoint>`              |
| `std::numeric_limits<FloatingPoint>::min_exponent10`  | `min_exponent10_v<FloatingPoint>`            |
| `std::numeric_limits<FloatingPoint>::max_exponent`    | `max_exponent_v<FloatingPoint>`              |
| `std::numeric_limits<FloatingPoint>::max_exponent10`  | `max_exponent10_v<FloatingPoint>`            |

Individual traits have `value` member constant that can be used with C++14 (e.g. `epsilon<float>::value`).

---

Individual traits are SFINAE-friendly, you can detect value presence/absence.

```C++
template <class T>
constexpr auto has_infinity(T)
    -> decltype(Kokkos::Experimental::infinity<T>::value, std::true_type{}) {
  return {};
}

constexpr std::false_type has_infinity(...) { return {}; }

template <class T>
KOKKOS_FUNCTION constexpr std::enable_if_t<has_infinity(T{}), T>
legacy_std_numeric_limits_infinity() {
  return Kokkos::Experimental::infinity<T>::value;
}

template <class T>
KOKKOS_FUNCTION constexpr std::enable_if_t<!has_infinity(T{}), T>
legacy_std_numeric_limits_infinity() {
  return T();
}
```

---
**See also**  
[Mathematical constants](mathematical-constants)  
[Common mathematical functions](mathematical-functions)  
