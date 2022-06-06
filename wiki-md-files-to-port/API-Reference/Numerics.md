# Numerics

**[Common mathematical functions](Mathematical-Functions)**  
The header `<Kokkos_MathematicalFunctions.hpp>` provides a consistent and
portable overload set for standard C library mathematical functions, such as
`fabs`, `sqrt`, and `sin`.

**Mathematical special functions**  
Work in progress

**[Mathematical constants](Mathematical-Constants)**  
The header `<Kokkos_MathematicalConstants.hpp>` is a backport of the C++20
standard library header `<numbers>` and provides several mathematical
constants, such as `pi` or `sqrt2`.

**[Numeric traits](Numeric-Traits)**  
The header `<Kokkos_NumericTraits.hpp>` implements a new facility that
is being added to the C++23 standard library and is intended as a
replacement for `std::numeric_limits`.
