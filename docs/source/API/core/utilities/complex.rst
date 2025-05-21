``complex``
===================

.. role:: cpp(code)
    :language: cpp

Defined in header ``<Kokkos_Complex.hpp>`` which is included from ``<Kokkos_Core.hpp>``

Description
-----------

``complex`` is a class template for representing and manipulating complex numbers.

* This is intended as a replacement for ``std::complex<T>``.
* Note: If ``z`` has type ``Kokkos::complex<T>``, casting such as ``reinterpret_cast<T(&)[2]>(z)`` leads to undefined behavior (this differs from ``std::complex``).

Interface
---------

.. cppkokkos:class:: template<class T> complex

  .. rubric:: Template Parameters

  :tparam T: The type of the real and imaginary components.

  * ``T`` must be a floating point type (``float``, ``double``, ``long double``) or an extended floating point type.

  * ``T`` cannot be ``const`` and/or ``volatile`` qualified.

  * Some types might not work with a specific backend (such as ``long double`` on CUDA or SYCL).

  .. rubric:: Public Types

  .. cppkokkos:type:: value_type = T

  .. rubric:: Constructors & Assignment Operators

  .. cppkokkos:function:: complex()

  Default constructor zero initializes the real and imaginary components.

  .. cppkokkos:function:: template<class U> complex(complex<U> z) noexcept


  Conversion constructor initializes the real component to ``static_cast<T>(z.real())`` and the imaginary component to ``static_cast<T>(z.imag())``.

  Constraints: ``U`` is convertible to ``T``.

  .. cppkokkos:function:: complex(std::complex<T> z) noexcept
  .. cppkokkos:function:: complex& operator=(std::complex<T> z) noexcept

  Implicit conversion from ``std::complex`` initializes the real component to ``z.real()`` and the imaginary component to ``z.imag()``.

  .. cppkokkos:function:: constexpr complex(T r) noexcept
  .. cppkokkos:function:: constexpr complex& operator=(T r) noexcept

  Initializes the real component to ``r`` and zero initializes the imaginary component.

  .. cppkokkos:function:: constexpr complex(T r, T i) noexcept

  Initializes the real component to ``r`` and the imaginary component to ``i``.

  .. deprecated:: 4.0.0
  .. cppkokkos:function:: template<class U> complex(const volatile complex<U>&) noexcept
  .. cppkokkos:function:: void operator=(const complex&) volatile noexcept
  .. cppkokkos:function:: volatile complex& operator=(const volatile complex&) volatile noexcept
  .. cppkokkos:function:: complex& operator=(const volatile complex&) noexcept
  .. cppkokkos:function:: void operator=(const volatile T&) noexcept
  .. cppkokkos:function:: void operator=(const T&) volatile noexcept

  Note: The assignment operators have templated implementations so as not to be copy assignment operators.

  .. rubric:: Public Member Functions

  .. cppkokkos:function:: operator std::complex<T>() const noexcept

  Conversion operator to ``std::complex``.

  .. cppkokkos:function:: constexpr T& real() noexcept
  .. cppkokkos:function:: constexpr T real() const noexcept

  :return: The value of the real component.

  .. cppkokkos:function:: constexpr void real(T r) noexcept

  Assigns ``r`` to the real component.

  .. cppkokkos:function:: constexpr T& imag() noexcept
  .. cppkokkos:function:: constexpr T imag() const noexcept

  :return: The value of the imaginary component.

  .. cppkokkos:function:: constexpr void imag(T i) noexcept

  Assigns ``i`` to the imaginary component.

  .. cppkokkos:function:: constexpr complex& operator+=(complex v) noexcept
  .. cppkokkos:function:: constexpr complex& operator+=(T v) noexcept

  Adds the complex value ``complex(v)`` to the complex value ``*this`` and stores the sum in ``*this``.

  .. cppkokkos:function:: constexpr complex& operator-=(complex v) noexcept
  .. cppkokkos:function:: constexpr complex& operator-=(T v) noexcept

  Subtracts the complex value ``complex(v)`` from the complex value ``*this`` and stores the difference in ``*this``.

  .. cppkokkos:function:: constexpr complex& operator*=(complex v) noexcept
  .. cppkokkos:function:: constexpr complex& operator*=(T v) noexcept

  Multiplies the complex value ``complex(v)`` by the complex value ``*this`` and stores the product in ``*this``.

  .. cppkokkos:function:: constexpr complex& operator/=(complex v) noexcept(noexcept(T{}/T{}))
  .. cppkokkos:function:: constexpr complex& operator/=(T v) noexcept(noexcept(T{}/T{}))

  Divides the complex value ``complex(v)`` into the complex value ``*this`` and stores the quotient in ``*this``.

  .. deprecated:: 4.0.0
  .. cppkokkos:function:: volatile T& real() volatile noexcept
  .. cppkokkos:function:: T real() const volatile noexcept
  .. cppkokkos:function:: volatile T& imag() volatile noexcept
  .. cppkokkos:function:: T imag() const volatile noexcept
  .. cppkokkos:function:: void operator+=(const volatile complex& v) volatile noexcept
  .. cppkokkos:function:: void operator+=(const volatile T& v) volatile noexcept
  .. cppkokkos:function:: void operator-=(const volatile complex& v) volatile noexcept
  .. cppkokkos:function:: void operator-=(const volatile T& v) volatile noexcept
  .. cppkokkos:function:: void operator*=(const volatile complex& v) volatile noexcept
  .. cppkokkos:function:: void operator*=(const volatile T& v) volatile noexcept
  .. cppkokkos:function:: void operator/=(const volatile complex& v) volatile noexcept(noexcept(T{}/T{}))
  .. cppkokkos:function:: void operator/=(const volatile T& v) volatile noexcept(noexcept(T{}/T{}))

