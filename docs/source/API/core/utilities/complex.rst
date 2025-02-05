``complex``
===================

.. role:: cpp(code)
    :language: cpp

Defined in header ``<Kokkos_Complex.hpp>`` which is included from ``<Kokkos_Core.hpp>``

Description
-----------

``complex`` is a class template for representing and manipulating complex numbers.

* This is intended as a replacement for ``std::complex<T>``.
* If ``z`` has type ``Kokkos::complex<T>``, casting such as ``reinterpret_cast<T(&)[2](z)`` lead to undefined behavior.  Note: This differs from ``std::complex``.

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

  .. deprecated:: 4.x.x
  .. cppkokkos:function:: void operator=(const complex&) volatile noexcept
  .. cppkokkos:function:: volatile complex& operator=(const volatile complex&) volatile noexcept
  .. cppkokkos:function:: complex& operator=(const volatile complex&) noexcept
  .. cppkokkos:function:: void operator=(const volatile T&) noexcept
  .. cppkokkos:function:: void operator=(const T&) volatile noexcept

  Note: These have templated implementations so as not to be copy assignment operators
