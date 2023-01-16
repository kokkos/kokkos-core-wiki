Common math functions
=====================

.. role::cpp(code)
    :language: cpp

.. role:: strike
    :class: strike

Motivating example (borrowed from https://llvm.org/docs/CompileCudaWithLLVM.html#standard-library-support)

.. code-block:: cpp

    // clang is OK with everything in this function.
    __device__ void test() {
        std::sin(0.); // nvcc - ok
        std::sin(0);  // nvcc - error, because no std::sin(int) override is available.
        sin(0);       // nvcc - same as above.

        sinf(0.);       // nvcc - ok
        std::sinf(0.);  // nvcc - no such function
    }

Kokkos' goal is to provide a consistent overload set that is available on host
and device and that follows practice from the C++ numerics library.

------------

.. _text: https://github.com/kokkos/kokkos/blob/develop/core/src/Kokkos_MathematicalFunctions.hpp

.. |text| replace:: ``<Kokkos_MathematicalFunctions.hpp>``

Defined in header |text|_ which is included from ``<Kokkos_Core.hpp>``

.. _text2: https://en.cppreference.com/w/cpp/numeric/math

.. |text2| replace:: standard C mathematical functions from ``<cmath>``

Provides most of the |text2|_, such as ``fabs``, ``sqrt``, and ``sin``.

Math functions are available in the ``Kokkos::`` namespace since version 3.7, in ``Kokkos::Experimental`` in previous versions.

Below is the synopsis for ``sqrt`` as an example of unary math function.

.. code-block:: cpp

    namespace Kokkos {  // (since 3.7)
        KOKKOS_FUNCTION float       sqrt ( float x );
        KOKKOS_FUNCTION float       sqrtf( float x );
        KOKKOS_FUNCTION double      sqrt ( double x );
                        long double sqrt ( long double x );
                        long double sqrtl( long double x );
        KOKKOS_FUNCTION double      sqrt ( IntegralType x );
    }

The function is overloaded for any argument of arithmetic type. Additional functions with ``f`` and ``l`` suffixes that work on ``float`` and ``long double`` respectively are also available.  Please note, that ``long double`` overloads are not available on the device.

See below the list of common mathematical functions supported. We refer the reader to cppreference.com for the synopsis of each individual function.

------------

:strike:`func` denotes functions that are currently not provided by Kokkos

``func*`` see notes below

.. _abs: https://en.cppreference.com/w/cpp/numeric/math/fabs

.. |abs| replace:: ``abs``

.. _fabs: https://en.cppreference.com/w/cpp/numeric/math/fabs

.. |fabs| replace:: ``fabs``

.. _fmod: https://en.cppreference.com/w/cpp/numeric/math/fmod

.. |fmod| replace:: ``fmod``

.. _remainder: https://en.cppreference.com/w/cpp/numeric/math/remainder

.. |remainder| replace:: ``remainder``

.. _remquo: https://en.cppreference.com/w/cpp/numeric/math/remquo

.. |remquo| replace:: <strike> ``remquo`` </strike>

.. _fma*: https://en.cppreference.com/w/cpp/numeric/math/fma

.. |fma*| replace:: ``fma*``

.. _fmax: https://en.cppreference.com/w/cpp/numeric/math/fmax

.. |fmax| replace:: ``fmax``

.. _fmin: https://en.cppreference.com/w/cpp/numeric/math/fmin

.. |fmin| replace:: ``fmin``

.. _fdim: https://en.cppreference.com/w/cpp/numeric/math/fdim

.. |fdim| replace:: ``fdim``

.. _nan: https://en.cppreference.com/w/cpp/numeric/math/nan

.. |nan| replace:: ``nan``

**Basic operations** |abs|_ |fabs|_ |fmod|_ |remainder|_ |remquo|_ |fma*|_ |fmax|_ |fmin|_ |fdim|_ |nan|_

**Exponential functions**

.. compo:

    .. _:
    .. || replace::
    ||_

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
    [`hypot*`](https://en.cppreference.com/w/cpp/numeric/math/hypot)

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
    [`round*`](https://en.cppreference.com/w/cpp/numeric/math/round)
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
    [`logb*`](https://en.cppreference.com/w/cpp/numeric/math/logb)
    [`nextafter*`](https://en.cppreference.com/w/cpp/numeric/math/nextafter)
    [<strike>`nexttoward`</strike>](https://en.cppreference.com/w/cpp/numeric/math/nexttoward)
    [`copysign*`](https://en.cppreference.com/w/cpp/numeric/math/copysign)

    **Classification and comparison**
    [<strike>`fpclassify`</strike>](https://en.cppreference.com/w/cpp/numeric/math/fpclassify)
    [`isfinite`](https://en.cppreference.com/w/cpp/numeric/math/isfinite)
    [`isinf`](https://en.cppreference.com/w/cpp/numeric/math/isinf)
    [`isnan`](https://en.cppreference.com/w/cpp/numeric/math/isnan)
    [<strike>`isnormal`</strike>](https://en.cppreference.com/w/cpp/numeric/math/isnormal)
    [`signbit*`](https://en.cppreference.com/w/cpp/numeric/math/signbit)
    [<strike>`isgreater`</strike>](https://en.cppreference.com/w/cpp/numeric/math/isgreater)
    [<strike>`isgreaterequal`</strike>](https://en.cppreference.com/w/cpp/numeric/math/isgreaterequal)
    [<strike>`isless`</strike>](https://en.cppreference.com/w/cpp/numeric/math/isless)
    [<strike>`islessequal`</strike>](https://en.cppreference.com/w/cpp/numeric/math/islessequal)
    [<strike>`islessgreater`</strike>](https://en.cppreference.com/w/cpp/numeric/math/islessgreater)
    [<strike>`isunordered`</strike>](https://en.cppreference.com/w/cpp/numeric/math/isunordered)

------------

Notes
-----

.. _openIssue: https://github.com/kokkos/kokkos/issues/new
.. |openIssue| replace:: **open an issue**

.. _issue4767: https://github.com/kokkos/kokkos/issues/4767
.. |issue4767| replace:: **Issue #4767**

* **Feel free to** |openIssue|_ **if you need one of the functions that is currently not implemented.** |issue4767|_ **is keeping track of these and has notes about implementability.**
* ``nearbyint`` is not available with the SYCL backend
* ``round``, ``logb``, ``nextafter``, ``copysign``, and ``signbit`` are available since version 3.7
* three-argument version of ``hypot`` is available since 4.0
* ``fma`` is available since 4.0

------------

See also
--------

`Mathematical constant <mathematical-constants.html>`_

`Numeric traits <numeric-traits.html>`_  