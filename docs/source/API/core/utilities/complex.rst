``Kokkos::complex``
===================

.. role:: cpp(code)
    :language: cpp

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

    Kokkos::complex<double> a,b;
    a.imag() = 5.0; a.real() = 1.0
    b = a;
    a += b;


Description
-----------

.. cpp:class:: template<class Scalar> complex

   |

   .. rubric:: Public Typedefs

   .. cpp:type:: value_type

      The scalar type of the real and the imaginary component.

   .. rubric:: Private Members

   .. cpp:member:: value_type im

   .. cpp:member:: value_type re

      Private data members representing the real and the imaginary parts.

   .. rubric:: Constructors

   .. cpp:function:: KOKKOS_INLINE_FUNCTION complex();

      Default constructor. Initializes the ``re`` and ``im`` with ``value_type()``.

   .. cpp:function:: KOKKOS_INLINE_FUNCTION complex(const complex& src);

      Copy constructor. Sets ``re = src.real()`` and ``im = src.imag()``.

   .. cpp:function:: template<class T> KOKKOS_INLINE_FUNCTION complex(const T& real);

      Constructor from a real number. Sets ``re = real`` and ``im = value_type()``.

   .. cpp:function:: template <class T1, class T2> KOKKOS_INLINE_FUNCTION complex(const T1& real, const T2& imag)

      Constructor from real numbers. Sets ``re = real`` and ``im = imag``.

   .. cpp:function:: template<class T> complex(const std::complex<T>& src);

      Copy constructor. Sets ``re = src.real()`` and ``im = src.imag()``.

   .. rubric:: Assignment and conversion

   .. cpp:function:: template<class T> KOKKOS_INLINE_FUNCTION complex<Scalar>& operator= (const complex<T>& src);

      Sets ``re = src.real()`` and ``im = src.imag()``.

   .. cpp:function:: template<class T> KOKKOS_INLINE_FUNCTION complex<Scalar>& operator= (const T& re);

      Sets ``re = src.real()`` and ``im = value_type()``.

   .. cpp:function:: template<class T> KOKKOS_INLINE_FUNCTION complex<Scalar>& operator= (const std::complex<T>& src);

      Sets ``re = src.real()`` and ``im = src.imag()``.

   .. cpp:function:: operator std::complex<value_type>() const;

      Returns ``std::complex<value_type>(re,im)``.

   .. rubric:: Functions

   .. cpp:function:: KOKKOS_INLINE_FUNCTION RealType& imag();

      Return ``im``.

   .. cpp:function:: KOKKOS_INLINE_FUNCTION RealType& real();

      Return ``re``.

   .. cpp:function:: KOKKOS_INLINE_FUNCTION const RealType imag() const;

      Return ``im``.

   .. cpp:function:: KOKKOS_INLINE_FUNCTION const RealType real() const;

      Return ``re``.

   .. cpp:function:: KOKKOS_INLINE_FUNCTION void imag(RealType v);

      Sets ``im = v``.

   .. cpp:function:: KOKKOS_INLINE_FUNCTION void real(RealType v);

      Sets ``re = v``.

   .. cpp:function:: template<class T>KOKKOS_INLINE_FUNCTION complex& operator += (const complex<T>& src);

      Executes ``re += src.real(); im += src.imag(); return *this;``

   .. cpp:function:: template<class T> complex& operator += (const std::complex<T>& src);

      Executes ``re += src.real(); im += src.imag(); return *this;``

   .. cpp:function:: template<class T> KOKKOS_INLINE_FUNCTION complex& operator += (const T& real);

      Executes ``re += real; return *this;``

   .. cpp:function:: template<class T> KOKKOS_INLINE_FUNCTION complex& operator -= (const complex<T>& src);

      Executes ``re -= src.real(); im -= src.imag(); return *this;``

   .. cpp:function:: template<class T> complex& operator -= (const std::complex<T>& src);

      Executes ``re -= src.real(); im -= src.imag(); return *this;``

   .. cpp:function:: template<class T> KOKKOS_INLINE_FUNCTION complex& operator -= (const T& real);

      Executes ``re -= real; return *this;``

   .. cpp:function:: template<class T> KOKKOS_INLINE_FUNCTION complex& operator *= (const complex<T>& src);

      Multiplies the current complex number with the complex number ``src``.

   .. cpp:function:: template<class T> complex& operator *= (const std::complex<T>& src);

      Multiplies the current complex number with the complex number ``src``.

   .. cpp:function:: template<class T> KOKKOS_INLINE_FUNCTION complex& operator *= (const T& real);

      Executes ``re *= real; im *= real; return *this;``

   .. cpp:function:: template<class T> KOKKOS_INLINE_FUNCTION complex& operator /= (const complex<T>& src);

      Divides the current complex number with the complex number ``src``.

   .. cpp:function:: template<class T> complex& operator /= (const std::complex<T>& src);

      Divides the current complex number with the complex number ``src``.

   .. cpp:function:: template<class T> KOKKOS_INLINE_FUNCTION complex& operator /= (const T& real);

      Executes ``re /= real; im /= real; return *this;``

   .. cpp:function:: template<class T> KOKKOS_INLINE_FUNCTION complex& operator == (const complex<T>& src);

      Returns ``re == src.real() && im == src.imag()``.

   .. cpp:function:: template<class T> complex& operator == (const std::complex<T>& src);

      Returns ``re == src.real() && im == src.imag()``.

   .. cpp:function:: template<class T> KOKKOS_INLINE_FUNCTION complex& operator == (const T& real);

      Returns ``re == src.real() && im == value_type()``.

   .. cpp:function:: template<class T> KOKKOS_INLINE_FUNCTION complex& operator != (const complex<T>& src);

      Returns ``re != src.real() || im != src.imag()``.

   .. cpp:function:: template<class T> complex& operator != (const std::complex<T>& src);

      Returns ``re != src.real() || im != src.imag()``.

   .. cpp:function:: template<class T> KOKKOS_INLINE_FUNCTION complex& operator != (const T& real);

      Returns ``re != src.real() || im != value_type()``.
