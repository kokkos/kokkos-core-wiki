Mathematical constants
======================

.. role::cpp(code)
    :language: cpp

.. _source_math_constants: https://github.com/kokkos/kokkos/blob/develop/core/src/Kokkos_MathematicalConstants.hpp

.. |source_math_constants| replace:: ``<Kokkos_MathematicalConstants.hpp>``

Defined in header |source_math_constants|_
which is included from ``<Kokkos_Core.hpp>``

.. _cpp_reference_numbers: https://en.cppreference.com/w/cpp/numeric/constants

.. |cpp_reference_numbers| replace:: ``<numbers>``

Provides all mathematical constants from |cpp_reference_numbers|_ (since C++20).

All constants are defined in the ``Kokkos::numbers::`` namespace since version 4.0, in ``Kokkos::Experimental`` in previous versions.

Variable Templates
------------------
The following are variable templates defined for standard floating-point types (``float``, ``double``, ``long double``).

.. list-table::
   :align: left
   :header-rows: 1

   * - Template name
     - Math symbol
     - Description
   * - ``e_v``
     - :math:`e`
     - Base of the natural logarithm
   * - ``log2e_v``
     - :math:`\log_{2}{e}`
     - Base-2 logarithm of e
   * - ``log10e_v``
     - :math:`\log_{10}{e}`
     - Base-10 logarithm of e
   * - ``pi_v``
     - :math:`\pi`
     - Ratio of a circle's circumference to its diameter
   * - ``inv_pi_v``
     - :math:`\frac{1}{\pi}`
     - Inverse of pi
   * - ``inv_sqrtpi_v``
     - :math:`\frac{1}{\sqrt{\pi}}`
     - Inverse of the square root of pi
   * - ``ln2_v``
     - :math:`\ln{2}`
     - Natural logarithm of 2
   * - ``ln10_v``
     - :math:`\ln{10}`
     - Natural logarithm of 10
   * - ``sqrt2_v``
     - :math:`\sqrt{2}`
     - Square root of 2
   * - ``sqrt3_v``
     - :math:`\sqrt{3}`
     - Square root of 3
   * - ``inv_sqrt3_v``
     - :math:`\frac{1}{\sqrt{3}}`
     - Inverse of the square root of 3
   * - ``egamma_v``
     - :math:`\gamma`
     - Euler-Mascheroni constant
   * - ``phi_v``
     - :math:`\varphi`
     - Golden ratio constant :math:`\frac{1+\sqrt{5}}{2}`

Convenience Constants (``double``)
----------------------------------
For each variable template listed above, Kokkos provides an ``inline constexpr
double`` constant without the ``_v`` suffix. These are shorthand for the
``double`` specialization.

* ``Kokkos::numbers::pi`` is equivalent to ``Kokkos::numbers::pi_v<double>``
* ``Kokkos::numbers::e`` is equivalent to ``Kokkos::numbers::e_v<double>``

------------

Notes
-----

.. _KnownIssues: ../../../known-issues.html#mathematical-constants

.. |KnownIssues| replace:: known issues

* The mathematical constants are available in ``Kokkos::Experimental::`` since Kokkos 3.6
* They were "promoted" to the ``Kokkos::numbers`` namespace in 4.0 and removed from ``Kokkos::Experimental::`` in 4.3
* Passing mathematical constants by reference or taking their address in device code is not supported by some toolchains and hence not portable.  (See |KnownIssues|_)
* Support for quadruple precision floating-point ``__float128`` can be enabled
  via ``-DKokkos_ENABLE_LIBQUADMATH=ON``.

------------

Example
-------

.. code-block:: cpp

    KOKKOS_FUNCTION void example() {
        constexpr auto pi = Kokkos::numbers::pi_v<float>;
        auto const x = Kokkos::sin(pi/6);
    }

------------

See also
--------

`Common mathematical functions <mathematical-functions.html>`_

`Numeric traits <numeric-traits.html>`_
