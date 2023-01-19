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

.. _exp: https://en.cppreference.com/w/cpp/numeric/math/exp

.. |exp| replace:: ``exp``

.. _exp2: https://en.cppreference.com/w/cpp/numeric/math/exp2

.. |exp2| replace:: ``exp2``

.. _expm1: https://en.cppreference.com/w/cpp/numeric/math/expm1

.. |expm1| replace:: ``expm1``

.. _log: https://en.cppreference.com/w/cpp/numeric/math/log

.. |log| replace:: ``log``

.. _log10: https://en.cppreference.com/w/cpp/numeric/math/log10

.. |log10| replace:: ``log10``

.. _log2: https://en.cppreference.com/w/cpp/numeric/math/log2

.. |log2| replace:: ``log2``

.. _log1p: https://en.cppreference.com/w/cpp/numeric/math/log1p

.. |log1p| replace:: ``log1p``

**Exponential functions** |exp|_ |exp2|_ |expm1|_ |log|_ |log10|_ |log2|_ |log1p|_

.. _pow: https://en.cppreference.com/w/cpp/numeric/math/pow

.. |pow| replace:: ``pow``

.. _sqrt: https://en.cppreference.com/w/cpp/numeric/math/sqrt

.. |sqrt| replace:: ``sqrt``

.. _cbrt: https://en.cppreference.com/w/cpp/numeric/math/cbrt

.. |cbrt| replace:: ``cbrt``

.. _hypot*: https://en.cppreference.com/w/cpp/numeric/math/hypot

.. |hypot*| replace:: ``hypot*``

**Power functions** |pow|_ |sqrt|_ |cbrt|_ |hypot*|_

.. _sin: https://en.cppreference.com/w/cpp/numeric/math/sin

.. |sin| replace:: ``sin``

.. _cos: https://en.cppreference.com/w/cpp/numeric/math/cos

.. |cos| replace:: ``cos``

.. _tan: https://en.cppreference.com/w/cpp/numeric/math/tan

.. |tan| replace:: ``tan``

.. _asin: https://en.cppreference.com/w/cpp/numeric/math/asin

.. |asin| replace:: ``asin``

.. _acos: https://en.cppreference.com/w/cpp/numeric/math/acos

.. |acos| replace:: ``acos``

.. _atan: https://en.cppreference.com/w/cpp/numeric/math/atan

.. |atan| replace:: ``atan``

.. _atan2: https://en.cppreference.com/w/cpp/numeric/math/atan2

.. |atan2| replace:: ``atan2``

**Trigonometric functions** |sin|_ |cos|_ |tan|_ |asin|_ |acos|_ |atan|_ |atan2|_

.. _sinh: https://en.cppreference.com/w/cpp/numeric/math/sinh

.. |sinh| replace:: ``sinh``

.. _cosh: https://en.cppreference.com/w/cpp/numeric/math/cosh

.. |cosh| replace:: ``cosh``

.. _tanh: https://en.cppreference.com/w/cpp/numeric/math/tanh

.. |tanh| replace:: ``tanh``

.. _asinh: https://en.cppreference.com/w/cpp/numeric/math/asinh

.. |asinh| replace:: ``asinh``

.. _acosh: https://en.cppreference.com/w/cpp/numeric/math/acosh

.. |acosh| replace:: ``acosh``

.. _atanh: https://en.cppreference.com/w/cpp/numeric/math/atanh

.. |atanh| replace:: ``atanh``

**Hyperbolic functions** |sinh|_ |cosh|_ |tanh|_ |asinh|_ |acosh|_ |atanh|_

.. _erf: https://en.cppreference.com/w/cpp/numeric/math/erf

.. |erf| replace:: ``erf``

.. _erfc: https://en.cppreference.com/w/cpp/numeric/math/erfc

.. |erfc| replace:: ``erfc``

.. _tgamma: https://en.cppreference.com/w/cpp/numeric/math/tgamma

.. |tgamma| replace:: ``tgamma``

.. _lgamma: https://en.cppreference.com/w/cpp/numeric/math/lgamma

.. |lgamma| replace:: ``lgamma``

**Error and gamma functions** |erf|_ |erfc|_ |tgamma|_ |lgamma|_

.. _ceil: https://en.cppreference.com/w/cpp/numeric/math/ceil

.. |ceil| replace:: ``ceil``

.. _floor: https://en.cppreference.com/w/cpp/numeric/math/floor

.. |floor| replace:: ``floor``

.. _trunc: https://en.cppreference.com/w/cpp/numeric/math/trunc

.. |trunc| replace:: ``trunc``

.. _round*: https://en.cppreference.com/w/cpp/numeric/math/round

.. |round*| replace:: ``round*``

.. _lround: https://en.cppreference.com/w/cpp/numeric/math/round

.. |lround| replace:: <strike> ``lround`` </strike>

.. _llround: https://en.cppreference.com/w/cpp/numeric/math/round

.. |llround| replace:: <strike> ``llround`` </strike>

.. _nearbyint*: https://en.cppreference.com/w/cpp/numeric/math/nearbyint

.. |nearbyint*| replace:: ``nearbyint*``

.. _rint: https://en.cppreference.com/w/cpp/numeric/math/rint

.. |rint| replace:: <strike> ``rint`` </strike>

.. _lrint: https://en.cppreference.com/w/cpp/numeric/math/rint

.. |lrint| replace:: <strike> ``lrint`` </strike>

.. _llrint: https://en.cppreference.com/w/cpp/numeric/math/rint

.. |llrint| replace:: <strike> ``llrint`` </strike>

**Nearest integer floating point operations** |ceil|_ |floor|_ |trunc|_ |round*|_ |lround|_ |llround|_ |nearbyint*|_ |rint|_ |lrint|_ |llrint|_

.. _frexp: https://en.cppreference.com/w/cpp/numeric/math/frexp

.. |frexp| replace:: <strike> ``frexp`` </strike>

.. _ldexp: https://en.cppreference.com/w/cpp/numeric/math/ldexp

.. |ldexp| replace:: <strike> ``ldexp`` </strike>

.. _modf: https://en.cppreference.com/w/cpp/numeric/math/modf

.. |modf| replace:: <strike> ``modf`` </strike>

.. _scalbn: https://en.cppreference.com/w/cpp/numeric/math/scalbn

.. |scalbn| replace:: <strike> ``scalbn`` </strike>

.. _scalbln: https://en.cppreference.com/w/cpp/numeric/math/scalbln

.. |scalbln| replace:: <strike> ``scalbln`` </strike>

.. _ilog: https://en.cppreference.com/w/cpp/numeric/math/ilog

.. |ilog| replace:: <strike> ``ilog`` </strike>

.. _logb*: https://en.cppreference.com/w/cpp/numeric/math/logb

.. |logb*| replace:: ``logb*``

.. _nextafter*: https://en.cppreference.com/w/cpp/numeric/math/nextafter 

.. |nextafter*| replace:: ``nextafter*``

.. _nexttoward: https://en.cppreference.com/w/cpp/numeric/math/nexttoward

.. |nexttoward| replace:: <strike> ``nexttoward`` </strike>

.. _copysign*: https://en.cppreference.com/w/cpp/numeric/math/copysign

.. |copysign*| replace:: ``copysign*``

**Floating point manipulation functions** |frexp|_ |ldexp|_ |modf|_ |scalbn|_ |scalbln|_ |ilog|_ |logb*|_ |nextafter*|_ |nexttoward|_ |copysign*|_

.. _fpclassify: https://en.cppreference.com/w/cpp/numeric/math/fpclassify

.. |fpclassify| replace:: <strike> ``fpclassify`` </strike>

.. _isfinite: https://en.cppreference.com/w/cpp/numeric/math/isfinite

.. |isfinite| replace:: ``isfinite``

.. _isinf: https://en.cppreference.com/w/cpp/numeric/math/isinf

.. |isinf| replace:: ``isinf``

.. _isnan: https://en.cppreference.com/w/cpp/numeric/math/isnan

.. |isnan| replace:: ``isnan``

.. _isnormal: https://en.cppreference.com/w/cpp/numeric/math/isnormal

.. |isnormal| replace:: <strike> ``isnormal`` </strike>

.. _signbit*: https://en.cppreference.com/w/cpp/numeric/math/signbit

.. |signbit*| replace:: ``signbit*``

.. _isgreater: https://en.cppreference.com/w/cpp/numeric/math/isgreater

.. |isgreater| replace:: <strike> ``isgreater`` </strike>

.. _isgreaterequal: https://en.cppreference.com/w/cpp/numeric/math/isgreaterequal

.. |isgreaterequal| replace:: <strike> ``isgreaterequal`` </strike>

.. _isless: https://en.cppreference.com/w/cpp/numeric/math/isless

.. |isless| replace:: <strike> ``isless`` </strike>

.. _islessequal: https://en.cppreference.com/w/cpp/numeric/math/islessequal

.. |islessequal| replace:: <strike> ``islessequal`` </strike>

.. _islessgreater: https://en.cppreference.com/w/cpp/numeric/math/islessgreater

.. |islessgreater| replace:: <strike> ``islessgreater`` </strike>

.. _isunordered: https://en.cppreference.com/w/cpp/numeric/math/isunordered

.. |isunordered| replace:: <strike> ``isunordered`` </strike>

**Classification and comparison** |fpclassify|_ |isfinite|_ |isinf|_ |isnan|_ |isnormal|_ |signbit*|_ |isgreater|_ |isgreaterequal|_ |isless|_ |islessequal|_ |islessgreater|_ |isunordered|_

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