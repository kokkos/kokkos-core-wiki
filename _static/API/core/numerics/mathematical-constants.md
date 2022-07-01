# Mathematical constants

Defined in header [`<Kokkos_MathematicalConstants.hpp>`](https://github.com/kokkos/kokkos/blob/develop/core/src/Kokkos_MathematicalConstants.hpp)
which is included from `<Kokkos_Core.hpp>` (since Kokkos 3.6)

Provides all mathematical constants from [`<numbers>`](https://en.cppreference.com/w/cpp/numeric/constants) (since C++20).

All constants are defined in the `Kokkos::Experimental` namespace.

**Mathematical constants**
`e`
`log2e`
`log10e`
`pi`
`inv_pi`
`inv_sqrtpi`
`ln2`
`ln10`
`sqrt2`
`sqrt3`
`inv_sqrt3`
`egamma`
`phi`


| Standard library                                      | Kokkos with C++20                            |
|-------------------------------------------------------|----------------------------------------------|
| `std::numbers::e`                                     | `Kokkos::Experimental::e`                    |
| `std::numbers::log2e`                                 | `Kokkos::Experimental::log2e`                |
| `std::numbers::log10e`                                | `Kokkos::Experimental::log10e`               |
| `std::numbers::pi`                                    | `Kokkos::Experimental::pi`                   |
| `std::numbers::inv_pi`                                | `Kokkos::Experimental::inv_pi`               |
| `std::numbers::inv_sqrtpi`                            | `Kokkos::Experimental::lnv_sqrtpi`           |
| `std::numbers::ln2`                                   | `Kokkos::Experimental::ln2`                  |
| `std::numbers::ln10`                                  | `Kokkos::Experimental::ln10`                 |
| `std::numbers::sqrt2`                                 | `Kokkos::Experimental::sqrt2`                |
| `std::numbers::sqrt3`                                 | `Kokkos::Experiemtnal::sqrt3`                |
| `std::numbers::inv_sqrt3`                             | `Kokkos::Experimental::inv_sqrt3`            |
| `std::numbers::egamma`                                | `Kokkos::Experimental::egamma`               |
| `std::numbers::phi`                                   | `Kokkos::Experimental::phi`                  |

Values of constants rely on `<numbers>`. Refer to (https://eel.is/c++draft/numbers#math.constants) for details.

---

```C++
KOKKOS_FUNCTION void example() {
  constexpr auto pi = Kokkos::Experimental::pi_v<float>;
  auto const x = Kokkos::Experimental::sin(pi/6);
}
```

---
**See also**  
[Common mathematical functions](mathematical-functions)  
[Numeric traits](numeric-traits)  
