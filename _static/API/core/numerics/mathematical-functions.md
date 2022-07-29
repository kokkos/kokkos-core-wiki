# Common math functions

Motivating example (borrowed from https://llvm.org/docs/CompileCudaWithLLVM.html#standard-library-support)
```C++
// clang is OK with everything in this function.
__device__ void test() {
  std::sin(0.); // nvcc - ok
  std::sin(0);  // nvcc - error, because no std::sin(int) override is available.
  sin(0);       // nvcc - same as above.

  sinf(0.);       // nvcc - ok
  std::sinf(0.);  // nvcc - no such function
}
```
Kokkos' goal is to provide a consistent overload set that is available on host
and device and that follows practice from the C++ numerics library.

---

Defined in
header [`<Kokkos_MathematicalFunctions.hpp>`](https://github.com/kokkos/kokkos/blob/develop/core/src/Kokkos_MathematicalFunctions.hpp)
which is included from `<Kokkos_Core.hpp>`

Provides most of the [standard C mathematical functions from `<cmath>`](https://en.cppreference.com/w/cpp/numeric/math), such as `fabs`, `sqrt`, and `sin`.

Math functions are available in the `Kokkos::` namespace since version 3.7, in `Kokkos::Experimental` in previous versions.

Below is the synopsis for `sqrt` as an example of unary math function.
```C++
namespace Kokkos {  // (since 3.7)
KOKKOS_FUNCTION float       sqrt ( float x );
KOKKOS_FUNCTION float       sqrtf( float x );
KOKKOS_FUNCTION double      sqrt ( double x );
                long double sqrt ( long double x );
                long double sqrtl( long double x );
KOKKOS_FUNCTION double      sqrt ( IntegralType x );
}
```
The function is overloaded for any argument of arithmetic type.  Additional
functions with `f` and `l` suffixes that work on `float` and `long double`
respectively are also available.  Please note, that `long double` overloads are
not available on the device.

See below the list of common mathematical functions supported.  We refer the
reader to cppreference.com for the synopsis of each individual function.

---

<strike>_`func`_</strike> denotes functions that are currently not provided by Kokkos  
_`func*`_ denotes functions not available with the SYCL backend  
_`func**`_ denotes functions available since version 3.7

**Basic operations**
[`abs`](https://en.cppreference.com/w/cpp/numeric/math/fabs)
[`fabs`](https://en.cppreference.com/w/cpp/numeric/math/fabs)
[`fmod`](https://en.cppreference.com/w/cpp/numeric/math/fmod)
[`remainder`](https://en.cppreference.com/w/cpp/numeric/math/remainder)
[<strike>`remquo`</strike>](https://en.cppreference.com/w/cpp/numeric/math/remquo)
[<strike>`fma`</strike>](https://en.cppreference.com/w/cpp/numeric/math/fma)
[`fmax`](https://en.cppreference.com/w/cpp/numeric/math/fmax)
[`fmin`](https://en.cppreference.com/w/cpp/numeric/math/fmin)
[`fdim`](https://en.cppreference.com/w/cpp/numeric/math/fdim)
[`nan`](https://en.cppreference.com/w/cpp/numeric/math/nan)

**Exponential functions**
[`exp`](https://en.cppreference.com/w/cpp/numeric/math/exp)
[`exp2`](https://en.cppreference.com/w/cpp/numeric/math/exp2)
[`expm1`](https://en.cppreference.com/w/cpp/numeric/math/expm1)
[`log`](https://en.cppreference.com/w/cpp/numeric/math/log)
[`log10`](https://en.cppreference.com/w/cpp/numeric/math/log10)
[`log2`](https://en.cppreference.com/w/cpp/numeric/math/log2)
[`log1p`](https://en.cppreference.com/w/cpp/numeric/math/log1p)

**Power functions**
[`pow`](https://en.cppreference.com/w/cpp/numeric/math/pow)
[`sqrt`](https://en.cppreference.com/w/cpp/numeric/math/sqrt)
[`cbrt`](https://en.cppreference.com/w/cpp/numeric/math/cbrt)
[`hypot`](https://en.cppreference.com/w/cpp/numeric/math/hypot)

**Trigonometric functions**
[`sin`](https://en.cppreference.com/w/cpp/numeric/math/sin)
[`cos`](https://en.cppreference.com/w/cpp/numeric/math/cos)
[`tan`](https://en.cppreference.com/w/cpp/numeric/math/tan)
[`asin`](https://en.cppreference.com/w/cpp/numeric/math/asin)
[`acos`](https://en.cppreference.com/w/cpp/numeric/math/acos)
[`atan`](https://en.cppreference.com/w/cpp/numeric/math/atan)
[`atan2`](https://en.cppreference.com/w/cpp/numeric/math/atan2)

**Hyperbolic functions**
[`sinh`](https://en.cppreference.com/w/cpp/numeric/math/sinh)
[`cosh`](https://en.cppreference.com/w/cpp/numeric/math/cosh)
[`tanh`](https://en.cppreference.com/w/cpp/numeric/math/tanh)
[`asinh`](https://en.cppreference.com/w/cpp/numeric/math/asinh)
[`acosh`](https://en.cppreference.com/w/cpp/numeric/math/acosh)
[`atanh`](https://en.cppreference.com/w/cpp/numeric/math/atanh)

**Error and gamma functions**
[`erf`](https://en.cppreference.com/w/cpp/numeric/math/erf)
[`erfc`](https://en.cppreference.com/w/cpp/numeric/math/erfc)
[`tgamma`](https://en.cppreference.com/w/cpp/numeric/math/tgamma)
[`lgamma`](https://en.cppreference.com/w/cpp/numeric/math/lgamma)

**Nearest integer floating point operations**
[`ceil`](https://en.cppreference.com/w/cpp/numeric/math/ceil)
[`floor`](https://en.cppreference.com/w/cpp/numeric/math/floor)
[`trunc`](https://en.cppreference.com/w/cpp/numeric/math/trunc)
[`round**`](https://en.cppreference.com/w/cpp/numeric/math/round)
[<strike>`lround`</strike>](https://en.cppreference.com/w/cpp/numeric/math/round)
[<strike>`llround`</strike>](https://en.cppreference.com/w/cpp/numeric/math/round)
[`nearbyint*`](https://en.cppreference.com/w/cpp/numeric/math/nearbyint)
[<strike>`rint`</strike>](https://en.cppreference.com/w/cpp/numeric/math/rint)
[<strike>`lrint`</strike>](https://en.cppreference.com/w/cpp/numeric/math/rint)
[<strike>`llrint`</strike>](https://en.cppreference.com/w/cpp/numeric/math/rint)

**Floating point manipulation functions**
[<strike>`frexp`</strike>](https://en.cppreference.com/w/cpp/numeric/math/frexp)
[<strike>`ldexp`</strike>](https://en.cppreference.com/w/cpp/numeric/math/ldexp)
[<strike>`modf`</strike>](https://en.cppreference.com/w/cpp/numeric/math/modf)
[<strike>`scalbn`</strike>](https://en.cppreference.com/w/cpp/numeric/math/scalbn)
[<strike>`scalbln`</strike>](https://en.cppreference.com/w/cpp/numeric/math/scalbln)
[<strike>`ilog`</strike>](https://en.cppreference.com/w/cpp/numeric/math/ilog)
[`logb**`](https://en.cppreference.com/w/cpp/numeric/math/logb)
[`nextafter**`](https://en.cppreference.com/w/cpp/numeric/math/nextafter)
[<strike>`nexttoward`</strike>](https://en.cppreference.com/w/cpp/numeric/math/nexttoward)
[`copysign**`](https://en.cppreference.com/w/cpp/numeric/math/copysign)

**Classification and comparison**
[<strike>`fpclassify`</strike>](https://en.cppreference.com/w/cpp/numeric/math/fpclassify)
[`isfinite`](https://en.cppreference.com/w/cpp/numeric/math/isfinite)
[`isinf`](https://en.cppreference.com/w/cpp/numeric/math/isinf)
[`isnan`](https://en.cppreference.com/w/cpp/numeric/math/isnan)
[<strike>`isnormal`</strike>](https://en.cppreference.com/w/cpp/numeric/math/isnormal)
[`signbit**`](https://en.cppreference.com/w/cpp/numeric/math/signbit)
[<strike>`isgreater`</strike>](https://en.cppreference.com/w/cpp/numeric/math/isgreater)
[<strike>`isgreaterequal`</strike>](https://en.cppreference.com/w/cpp/numeric/math/isgreaterequal)
[<strike>`isless`</strike>](https://en.cppreference.com/w/cpp/numeric/math/isless)
[<strike>`islessequal`</strike>](https://en.cppreference.com/w/cpp/numeric/math/islessequal)
[<strike>`islessgreater`</strike>](https://en.cppreference.com/w/cpp/numeric/math/islessgreater)
[<strike>`isunordered`</strike>](https://en.cppreference.com/w/cpp/numeric/math/isunordered)

---

**NOTE** Feel free to [open an issue](https://github.com/kokkos/kokkos/issues/new) if you need one of the functions that is currently not implemented.  [Issue #4767](https://github.com/kokkos/kokkos/issues/4767) is keeping track of these and has notes about implementability.

---
**See also**  
[Mathematical constants](mathematical-constants)  
[Numeric traits](numeric-traits)  
