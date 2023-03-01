``Kokkos::complex``
===================

.. role:: cppkokkos(code)
    :language: cppkokkos

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

.. cppkokkos:class:: template<class Scalar> complex

   |

   .. rubric:: Public Typedefs

   .. cppkokkos:type:: value_type

      The scalar type of the real and the imaginary component.

   .. rubric:: Constructors

   .. cppkokkos:kokkosinlinefunction:: complex();

      Default constructor. Initializes the ``re`` and ``im`` with ``value_type()``.

   .. cppkokkos:kokkosinlinefunction:: complex(const complex& src);

      Copy constructor. Sets ``re = src.real()`` and ``im = src.imag()``.

   .. cppkokkos:kokkosinlinefunction:: template<class T> complex(const T& real);

      Constructor from a real number. Sets ``re = real`` and ``im = value_type()``.

   .. cppkokkos:kokkosinlinefunction:: template <class T1, class T2> complex(const T1& real, const T2& imag)

      Constructor from real numbers. Sets ``re = real`` and ``im = imag``.

   .. cppkokkos:function:: template<class T> complex(const std::complex<T>& src);

      Copy constructor. Sets ``re = src.real()`` and ``im = src.imag()``.

   .. rubric:: Assignment and conversion

   .. cppkokkos:kokkosinlinefunction:: template<class T> complex<Scalar>& operator= (const complex<T>& src);

      Sets ``re = src.real()`` and ``im = src.imag()``.

   .. cppkokkos:kokkosinlinefunction:: template<class T> complex<Scalar>& operator= (const T& re);

      Sets ``re = src.real()`` and ``im = value_type()``.

   .. cppkokkos:kokkosinlinefunction:: template<class T> complex<Scalar>& operator= (const std::complex<T>& src);

      Sets ``re = src.real()`` and ``im = src.imag()``.

   .. cppkokkos:function:: operator std::complex<value_type>() const;

      Returns ``std::complex<value_type>(re,im)``.

   .. rubric:: Functions

   .. cppkokkos:kokkosinlinefunction:: RealType& imag();

      Return ``im``.

   .. cppkokkos:kokkosinlinefunction:: RealType& real();

      Return ``re``.

   .. cppkokkos:kokkosinlinefunction:: const RealType imag() const;

      Return ``im``.

   .. cppkokkos:kokkosinlinefunction:: const RealType real() const;

      Return ``re``.

   .. cppkokkos:kokkosinlinefunction:: void imag(RealType v);

      Sets ``im = v``.

   .. cppkokkos:kokkosinlinefunction:: void real(RealType v);

      Sets ``re = v``.

   .. cppkokkos:kokkosinlinefunction:: template<class T> complex& operator += (const complex<T>& src);

      Executes ``re += src.real(); im += src.imag(); return *this;``

   .. cppkokkos:function:: template<class T> complex& operator += (const std::complex<T>& src);

      Executes ``re += src.real(); im += src.imag(); return *this;``

   .. cppkokkos:kokkosinlinefunction:: template<class T> complex& operator += (const T& real);

      Executes ``re += real; return *this;``

   .. cppkokkos:kokkosinlinefunction:: template<class T> complex& operator -= (const complex<T>& src);

      Executes ``re -= src.real(); im -= src.imag(); return *this;``

   .. cppkokkos:function:: template<class T> complex& operator -= (const std::complex<T>& src);

      Executes ``re -= src.real(); im -= src.imag(); return *this;``

   .. cppkokkos:kokkosinlinefunction:: template<class T> complex& operator -= (const T& real);

      Executes ``re -= real; return *this;``

   .. cppkokkos:kokkosinlinefunction:: template<class T> complex& operator *= (const complex<T>& src);

      Multiplies the current complex number with the complex number ``src``.

   .. cppkokkos:function:: template<class T> complex& operator *= (const std::complex<T>& src);

      Multiplies the current complex number with the complex number ``src``.

   .. cppkokkos:kokkosinlinefunction:: template<class T> complex& operator *= (const T& real);

      Executes ``re *= real; im *= real; return *this;``

   .. cppkokkos:kokkosinlinefunction:: template<class T> complex& operator /= (const complex<T>& src);

      Divides the current complex number with the complex number ``src``.

   .. cppkokkos:function:: template<class T> complex& operator /= (const std::complex<T>& src);

      Divides the current complex number with the complex number ``src``.

   .. cppkokkos:kokkosinlinefunction:: template<class T> complex& operator /= (const T& real);

      Executes ``re /= real; im /= real; return *this;``

   .. cppkokkos:kokkosinlinefunction:: template<class T> complex& operator == (const complex<T>& src);

      Returns ``re == src.real() && im == src.imag()``.

   .. cppkokkos:function:: template<class T> complex& operator == (const std::complex<T>& src);

      Returns ``re == src.real() && im == src.imag()``.

   .. cppkokkos:kokkosinlinefunction:: template<class T> complex& operator == (const T& real);

      Returns ``re == src.real() && im == value_type()``.

   .. cppkokkos:kokkosinlinefunction:: template<class T> complex& operator != (const complex<T>& src);

      Returns ``re != src.real() || im != src.imag()``.

   .. cppkokkos:function:: template<class T> complex& operator != (const std::complex<T>& src);

      Returns ``re != src.real() || im != src.imag()``.

   .. cppkokkos:kokkosinlinefunction:: template<class T> complex& operator != (const T& real);

      Returns ``re != src.real() || im != value_type()``.
