# Mathematical constants

Defined in header [`<Kokkos_MathematicalConstants.hpp>`](https://github.com/kokkos/kokkos/blob/develop/core/src/Kokkos_MathematicalConstants.hpp)
which is included from `<Kokkos_Core.hpp>`

Provides all mathematical constants from [`<numbers>`](https://en.cppreference.com/w/cpp/numeric/constants) (since C++20).

All constants are defined in the `Kokkos::numbers::` namespace since version 4.0, in `Kokkos::Experimental` in previous versions.

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

---

## Notes
* The mathematical constants are available in `Kokkos::Experimental::` since Kokkos 3.6
* They were "promoted" to the `Kokkos::numbers` namespace in 4.0

---

## Example

```C++
KOKKOS_FUNCTION void example() {
  constexpr auto pi = Kokkos::numbers::pi_v<float>;
  auto const x = Kokkos::sin(pi/6);
}
```

---

## See also
[Common mathematical functions](mathematical-functions)  
[Numeric traits](numeric-traits)  
