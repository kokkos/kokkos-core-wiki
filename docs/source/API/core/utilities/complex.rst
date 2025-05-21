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

.. cpp:class:: template<class T> complex

  .. rubric:: Template Parameters

  :tparam T: The type of the real and imaginary components.

  * ``T`` must be a floating point type (``float``, ``double``, ``long double``) or an extended floating point type.

  * ``T`` cannot be ``const`` and/or ``volatile`` qualified.

  * Some types might not work with a specific backend (such as ``long double`` on CUDA or SYCL).

  .. rubric:: Public Types

  .. cpp:type:: value_type = T

  .. rubric:: Constructors & Assignment Operators

  .. cpp:function:: complex()

  Default constructor zero initializes the real and imaginary components.

  .. cpp:function:: template<class U> complex(complex<U> z) noexcept


  Conversion constructor initializes the real component to ``static_cast<T>(z.real())`` and the imaginary component to ``static_cast<T>(z.imag())``.

  Constraints: ``U`` is convertible to ``T``.

  .. cpp:function:: complex(std::complex<T> z) noexcept
  .. cpp:function:: complex& operator=(std::complex<T> z) noexcept

  Implicit conversion from ``std::complex`` initializes the real component to ``z.real()`` and the imaginary component to ``z.imag()``.

  .. cpp:function:: constexpr complex(T r) noexcept
  .. cpp:function:: constexpr complex& operator=(T r) noexcept

  Initializes the real component to ``r`` and zero initializes the imaginary component.

  .. cpp:function:: constexpr complex(T r, T i) noexcept

  Initializes the real component to ``r`` and the imaginary component to ``i``.

  .. deprecated:: 4.0.0
  .. cpp:function:: template<class U> complex(const volatile complex<U>&) noexcept
  .. cpp:function:: void operator=(const complex&) volatile noexcept
  .. cpp:function:: volatile complex& operator=(const volatile complex&) volatile noexcept
  .. cpp:function:: complex& operator=(const volatile complex&) noexcept
  .. cpp:function:: void operator=(const volatile T&) noexcept
  .. cpp:function:: void operator=(const T&) volatile noexcept

  Note: The assignment operators have templated implementations so as not to be copy assignment operators.

  .. rubric:: Public Member Functions

  .. cpp:function:: operator std::complex<T>() const noexcept

  Conversion operator to ``std::complex``.

  .. cpp:function:: constexpr T& real() noexcept
  .. cpp:function:: constexpr T real() const noexcept

  :return: The value of the real component.

  .. cpp:function:: constexpr void real(T r) noexcept

  Assigns ``r`` to the real component.

  .. cpp:function:: constexpr T& imag() noexcept
  .. cpp:function:: constexpr T imag() const noexcept

  :return: The value of the imaginary component.

  .. cpp:function:: constexpr void imag(T i) noexcept

  Assigns ``i`` to the imaginary component.

  .. cpp:function:: constexpr complex& operator+=(complex v) noexcept
  .. cpp:function:: constexpr complex& operator+=(T v) noexcept

  Adds the complex value ``complex(v)`` to the complex value ``*this`` and stores the sum in ``*this``.

  .. cpp:function:: constexpr complex& operator-=(complex v) noexcept
  .. cpp:function:: constexpr complex& operator-=(T v) noexcept

  Subtracts the complex value ``complex(v)`` from the complex value ``*this`` and stores the difference in ``*this``.

  .. cpp:function:: constexpr complex& operator*=(complex v) noexcept
  .. cpp:function:: constexpr complex& operator*=(T v) noexcept

  Multiplies the complex value ``complex(v)`` by the complex value ``*this`` and stores the product in ``*this``.

  .. cpp:function:: constexpr complex& operator/=(complex v) noexcept(noexcept(T{}/T{}))
  .. cpp:function:: constexpr complex& operator/=(T v) noexcept(noexcept(T{}/T{}))

  Divides the complex value ``complex(v)`` into the complex value ``*this`` and stores the quotient in ``*this``.

  .. deprecated:: 4.0.0
  .. cpp:function:: volatile T& real() volatile noexcept
  .. cpp:function:: T real() const volatile noexcept
  .. cpp:function:: volatile T& imag() volatile noexcept
  .. cpp:function:: T imag() const volatile noexcept
  .. cpp:function:: void operator+=(const volatile complex& v) volatile noexcept
  .. cpp:function:: void operator+=(const volatile T& v) volatile noexcept
  .. cpp:function:: void operator-=(const volatile complex& v) volatile noexcept
  .. cpp:function:: void operator-=(const volatile T& v) volatile noexcept
  .. cpp:function:: void operator*=(const volatile complex& v) volatile noexcept
  .. cpp:function:: void operator*=(const volatile T& v) volatile noexcept
  .. cpp:function:: void operator/=(const volatile complex& v) volatile noexcept(noexcept(T{}/T{}))
  .. cpp:function:: void operator/=(const volatile T& v) volatile noexcept(noexcept(T{}/T{}))

Non-Member Functions
____________________

  .. cpp:function:: template<typename RealType1, typename RealType2> bool operator==(complex<RealType1> x, complex<RealType2> y) noexcept
  .. cpp:function:: template<typename RealType1, typename RealType2> bool operator==(complex<RealType1> x, RealType2 y) noexcept
  .. cpp:function:: template<typename RealType1, typename RealType2> bool operator==(RealType1 x, complex<RealType2> y) noexcept
  .. cpp:function:: template<typename RealType1, typename RealType2> bool operator==(complex<RealType1> x, std::complex<RealType2> y) noexcept
  .. cpp:function:: template<typename RealType1, typename RealType2> bool operator==(std::complex<RealType1> x, complex<RealType2> y) noexcept

  :return: ``true`` if and only if the real component of ``complex(x)`` equals the real component of ``complex(y)`` and the imaginary component of ``complex(x)`` equals the imaginary component of ``complex(y)``.

  .. cpp:function:: template<typename RealType1, typename RealType2> bool operator!=(complex<RealType1> x, complex<RealType2> y) noexcept
  .. cpp:function:: template<typename RealType1, typename RealType2> bool operator!=(complex<RealType1> x, RealType2 y) noexcept
  .. cpp:function:: template<typename RealType1, typename RealType2> bool operator!=(RealType1 x, complex<RealType2> y) noexcept
  .. cpp:function:: template<typename RealType1, typename RealType2> bool operator!=(complex<RealType1> x, std::complex<RealType2> y) noexcept
  .. cpp:function:: template<typename RealType1, typename RealType2> bool operator!=(std::complex<RealType1> x, complex<RealType2> y) noexcept

  :return: ``!(x == y)``

  .. cpp:function:: template<size_t I, typename RealType> constexpr RealType& get(complex<RealType>& z) noexcept
  .. cpp:function:: template<size_t I, typename RealType> constexpr RealType&& get(complex<RealType>&& z) noexcept
  .. cpp:function:: template<size_t I, typename RealType> constexpr const RealType& get(const complex<RealType>& z) noexcept
  .. cpp:function:: template<size_t I, typename RealType> constexpr const RealType&& get(complex<RealType>&& z) noexcept

  Tuple protocol / structured binding support.

  :return: if ``I`` == 0 returns a reference to the real component of ``z``;
           if ``I`` == 1 returns a reference to the imaginary component of ``z``.



