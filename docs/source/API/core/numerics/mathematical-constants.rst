Mathematical constants
======================

.. role::cpp(code)
    :language: cpp

.. _text: https://github.com/kokkos/kokkos/blob/develop/core/src/Kokkos_MathematicalConstants.hpp

.. |text| replace:: ``<Kokkos_MathematicalConstants.hpp>``

Defined in header |text|_
which is included from ``<Kokkos_Core.hpp>``

.. _text2: https://en.cppreference.com/w/cpp/numeric/constants

.. |text2| replace:: ``<numbers>``

Provides all mathematical constants from |text2|_ (since C++20).

All constants are defined in the ``Kokkos::numbers::`` namespace since version 4.0, in ``Kokkos::Experimental`` in previous versions.

**Mathematical constants**

``e``
``log2e``
``log10e``
``pi``
``inv_pi``
``inv_sqrtpi``
``ln2``
``ln10``
``sqrt2``
``sqrt3``
``inv_sqrt3``
``egamma``
``phi``

------------

Notes
-----

.. _KnownIssues: ../../../known-issues.html#mathematical-constants

.. |KnownIssues| replace:: known issues

* The mathematical constants are available in ``Kokkos::Experimental::`` since Kokkos 3.6
* They were "promoted" to the ``Kokkos::numbers`` namespace in 4.0
* Passing mathematical constants by reference or taking their address in device code is not supported by some toolchains and hence not portable.  (See |KnownIssues|_)

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
