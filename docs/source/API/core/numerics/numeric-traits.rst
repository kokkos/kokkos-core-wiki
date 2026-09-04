Numeric traits
==============

.. role::cpp(code)
    :language: cpp

.. _source_numeric_traits: https://github.com/kokkos/kokkos/blob/5.2.0/core/src/Kokkos_NumericTraits.hpp

.. |source_numeric_traits| replace:: ``<Kokkos_NumericTraits.hpp>``

Defined in header |source_numeric_traits|_
which is included from ``<Kokkos_Core.hpp>``

.. note::
   Numeric traits implement a facility originally proposed for the
   C++ standard library in `P1841
   <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p1841r2.pdf>`__,
   following the clarifications made in `P2551
   <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p2551r2.pdf>`__.
   Neither proposal has been adopted into the C++ standard, and there is no
   standard library equivalent to switch to at this time. The ``Kokkos``
   namespace traits are intended for use in device code, where
   ``std::numeric_limits`` may not be usable.

Usage
-----

.. code-block:: cpp

   constexpr auto inf = Kokkos::infinity<float>::value;
   auto x = Kokkos::finite_min_v<T>;

.. _cpp_reference_numeric_limits: https://en.cppreference.com/w/cpp/types/numeric_limits

.. |cpp_reference_numeric_limits| replace:: ``std::numeric_limits``

Provides a replacement for |cpp_reference_numeric_limits|_ from the standard
library header ``<limits>`` that also works in device code, breaking the
monolithic ``numeric_limits`` class template apart into individual trait
templates.

Numeric traits are defined in the ``Kokkos`` namespace since Kokkos 5.2, and
in the ``Kokkos::Experimental`` namespace for earlier versions.

Individual Traits
-----------------
The following traits are class templates with a static constexpr ``value``
member.
Each trait is only defined for the argument type(s) for which it is
meaningful (`floating-point` or `arithmetic` in the tables below).

Numeric Distinguished Value Traits
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :align: left
   :header-rows: 1

   * - Trait name
     - Description
     - Valid for
   * - ``infinity``
     - Value representing positive infinity
     - floating-point types
   * - ``finite_min``
     - Lowest finite value
     - arithmetic types
   * - ``finite_max``
     - Largest finite value
     - arithmetic types
   * - ``epsilon``
     - Difference between 1 and the next representable value greater than 1
     - floating-point types
   * - ``round_error``
     - Maximum rounding error
     - floating-point types
   * - ``norm_min``
     - Smallest positive normalized value
     - floating-point types
   * - ``denorm_min``
     - Smallest positive subnormal value, or smallest positive normalized
       value if subnormals are not supported
     - floating-point types
   * - ``quiet_NaN``
     - A quiet (non-signaling) NaN value
     - floating-point types
   * - ``signaling_NaN``
     - A signaling NaN value
     - floating-point types

Numeric Characteristics Traits
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :align: left
   :header-rows: 1

   * - Trait name
     - Description
     - Valid for
   * - ``digits``
     - Number of radix digits that can be represented without change
     - arithmetic types
   * - ``digits10``
     - Number of decimal digits that can be represented without change
     - arithmetic types
   * - ``max_digits10``
     - Number of decimal digits needed to represent all distinct values
     - floating-point types
   * - ``radix``
     - Base (radix) used by the representation
     - arithmetic types
   * - ``min_exponent``
     - Lowest negative exponent such that ``radix`` raised to that power is a
       normalized value
     - floating-point types
   * - ``min_exponent10``
     - Lowest negative exponent such that 10 raised to that power is a
       normalized value
     - floating-point types
   * - ``max_exponent``
     - Largest positive exponent such that ``radix`` raised to that power is
       a representable finite value
     - floating-point types
   * - ``max_exponent10``
     - Largest positive exponent such that 10 raised to that power is a
       representable finite value
     - floating-point types

Variable Templates
------------------
For each trait listed above, Kokkos provides a variable template with a
``_v`` suffix. These are shorthand for the trait's ``value`` member.

* ``Kokkos::epsilon_v<T>`` is equivalent to ``Kokkos::epsilon<T>::value``
* ``Kokkos::infinity_v<T>`` is equivalent to ``Kokkos::infinity<T>::value``

Standard Library Equivalence
----------------------------
Each trait mirrors a corresponding member of |cpp_reference_numeric_limits|_.

.. _numlim_infinity: https://en.cppreference.com/w/cpp/types/numeric_limits/infinity
.. _numlim_lowest: https://en.cppreference.com/w/cpp/types/numeric_limits/lowest
.. _numlim_max: https://en.cppreference.com/w/cpp/types/numeric_limits/max
.. _numlim_epsilon: https://en.cppreference.com/w/cpp/types/numeric_limits/epsilon
.. _numlim_round_error: https://en.cppreference.com/w/cpp/types/numeric_limits/round_error
.. _numlim_min: https://en.cppreference.com/w/cpp/types/numeric_limits/min
.. _numlim_denorm_min: https://en.cppreference.com/w/cpp/types/numeric_limits/denorm_min
.. _numlim_quiet_NaN: https://en.cppreference.com/w/cpp/types/numeric_limits/quiet_NaN
.. _numlim_signaling_NaN: https://en.cppreference.com/w/cpp/types/numeric_limits/signaling_NaN
.. _numlim_digits: https://en.cppreference.com/w/cpp/types/numeric_limits/digits
.. _numlim_digits10: https://en.cppreference.com/w/cpp/types/numeric_limits/digits10
.. _numlim_max_digits10: https://en.cppreference.com/w/cpp/types/numeric_limits/max_digits10
.. _numlim_radix: https://en.cppreference.com/w/cpp/types/numeric_limits/radix
.. _numlim_min_exponent: https://en.cppreference.com/w/cpp/types/numeric_limits/min_exponent
.. _numlim_min_exponent10: https://en.cppreference.com/w/cpp/types/numeric_limits/min_exponent10
.. _numlim_max_exponent: https://en.cppreference.com/w/cpp/types/numeric_limits/max_exponent
.. _numlim_max_exponent10: https://en.cppreference.com/w/cpp/types/numeric_limits/max_exponent10

.. |numlim_infinity| replace:: ``std::numeric_limits<FloatingPoint>::infinity()``
.. |numlim_lowest| replace:: ``std::numeric_limits<Arithmetic>::lowest()``
.. |numlim_max| replace:: ``std::numeric_limits<Arithmetic>::max()``
.. |numlim_epsilon| replace:: ``std::numeric_limits<FloatingPoint>::epsilon()``
.. |numlim_round_error| replace:: ``std::numeric_limits<FloatingPoint>::round_error()``
.. |numlim_min| replace:: ``std::numeric_limits<FloatingPoint>::min()``
.. |numlim_denorm_min| replace:: ``std::numeric_limits<FloatingPoint>::denorm_min()``
.. |numlim_quiet_NaN| replace:: ``std::numeric_limits<FloatingPoint>::quiet_NaN()``
.. |numlim_signaling_NaN| replace:: ``std::numeric_limits<FloatingPoint>::signaling_NaN()``
.. |numlim_digits| replace:: ``std::numeric_limits<Arithmetic>::digits``
.. |numlim_digits10| replace:: ``std::numeric_limits<Arithmetic>::digits10``
.. |numlim_max_digits10| replace:: ``std::numeric_limits<FloatingPoint>::max_digits10``
.. |numlim_radix| replace:: ``std::numeric_limits<Arithmetic>::radix``
.. |numlim_min_exponent| replace:: ``std::numeric_limits<FloatingPoint>::min_exponent``
.. |numlim_min_exponent10| replace:: ``std::numeric_limits<FloatingPoint>::min_exponent10``
.. |numlim_max_exponent| replace:: ``std::numeric_limits<FloatingPoint>::max_exponent``
.. |numlim_max_exponent10| replace:: ``std::numeric_limits<FloatingPoint>::max_exponent10``

.. list-table::
   :align: left
   :header-rows: 1

   * - Trait name
     - Equivalent to
   * - ``infinity``
     - |numlim_infinity|_
   * - ``finite_min``
     - |numlim_lowest|_
   * - ``finite_max``
     - |numlim_max|_
   * - ``epsilon``
     - |numlim_epsilon|_
   * - ``round_error``
     - |numlim_round_error|_
   * - ``norm_min``
     - |numlim_min|_
   * - ``denorm_min``
     - |numlim_denorm_min|_
   * - ``quiet_NaN``
     - |numlim_quiet_NaN|_
   * - ``signaling_NaN``
     - |numlim_signaling_NaN|_
   * - ``digits``
     - |numlim_digits|_
   * - ``digits10``
     - |numlim_digits10|_
   * - ``max_digits10``
     - |numlim_max_digits10|_
   * - ``radix``
     - |numlim_radix|_
   * - ``min_exponent``
     - |numlim_min_exponent|_
   * - ``min_exponent10``
     - |numlim_min_exponent10|_
   * - ``max_exponent``
     - |numlim_max_exponent|_
   * - ``max_exponent10``
     - |numlim_max_exponent10|_

.. note::
   ``Arithmetic`` denotes any integral or floating-point type (i.e. any type
   for which ``std::is_arithmetic_v`` is ``true``), while ``FloatingPoint``
   denotes any floating-point type (i.e. any type for which
   ``std::is_floating_point_v`` is ``true``). These match the "Valid for"
   column in the tables above: the Kokkos trait is only specialized for
   those types. This is a deliberate difference from
   |cpp_reference_numeric_limits|_, which is defined for every type and
   silently returns a meaningless value outside its intended domain (e.g.
   ``std::numeric_limits<int>::infinity()`` returns ``0``), a source of bugs
   that the Kokkos traits are designed to avoid.

------------

Notes
-----

.. _KnownIssues: ../../../known-issues.html#mathematical-constants-and-numeric-traits

.. |KnownIssues| replace:: known issues

.. important::
   **Portability:** Passing numeric traits by reference or taking their
   address in device code is not supported by some toolchains and hence not
   portable.  (See |KnownIssues|_)

.. note::
   **Detecting specialization:** Because each trait is only specialized for
   the types listed in its "Valid for" column, generic code can detect
   whether a specialization exists for a given type before using it, rather
   than silently falling back to a meaningless value the way
   |cpp_reference_numeric_limits|_ does.

   With C++14, the minimum standard required when numeric traits were first
   introduced in Kokkos, this detection takes the form of expression SFINAE
   against the trait's ``value`` member, e.g.:

   .. code-block:: cpp

       template <class T>
       constexpr auto has_infinity(T)
               -> decltype(Kokkos::infinity<T>::value, std::true_type{}) {
           return {};
       }

       constexpr std::false_type has_infinity(...) { return {}; }

   The Example section below builds on ``has_infinity`` to implement a
   device-compatible replacement for
   ``std::numeric_limits<T>::infinity()``.

   With C++20, a ``requires`` clause combined with ``if constexpr`` offers a
   more direct alternative, without needing a separate detection function.

------------

Example
-------

.. code-block:: cpp

    template <class T>
    KOKKOS_FUNCTION constexpr std::enable_if_t<has_infinity(T{}), T>
    legacy_std_numeric_limits_infinity() {
        return Kokkos::infinity<T>::value;
    }

    template <class T>
    KOKKOS_FUNCTION constexpr std::enable_if_t<!has_infinity(T{}), T>
    legacy_std_numeric_limits_infinity() {
        return T();
    }

With C++20:

.. code-block:: cpp

    template <class T>
    KOKKOS_FUNCTION constexpr T legacy_std_numeric_limits_infinity() {
        if constexpr (requires { Kokkos::infinity<T>::value; }) {
            return Kokkos::infinity_v<T>;
        } else {
            return T();
        }
    }

------------

See also
--------

.. seealso::
   `Mathematical constants <mathematical-constants.html>`_

   `Common mathematical functions <mathematical-functions.html>`_
