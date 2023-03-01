Numerics
=========

.. toctree::
   :maxdepth: 1

   numerics/mathematical-constants.md

The header ``<Kokkos_MathematicalConstants.hpp>`` is a backport of the C++20 standard library header ``<numbers>`` and
provides several mathematical constants, such as ``pi`` or ``sqrt2``.

.. toctree::
   :maxdepth: 1

   numerics/mathematical-functions.md

The header ``<Kokkos_MathematicalFunctions.hpp>`` provides a consistent and portable overload set for standard C
library mathematical functions, such as ``fabs``, ``sqrt``, and ``sin``.

.. toctree::
   :maxdepth: 1

   numerics/numeric-traits.md

The header ``<Kokkos_NumericTraits.hpp>`` implements a new facility that is being added to the C++23 standard library and
is intended as a replacement for ``std::numeric_limits``.

.. toctree::
   :maxdepth: 1

   numerics/bit-manipulation.md

The header ``<Kokkos_BitManipulation.hpp>`` is a backport of the C++20 standard library header ``<bit>`` and
provides several function templates to access, manipulate, and process individual bits and bit sequences.
