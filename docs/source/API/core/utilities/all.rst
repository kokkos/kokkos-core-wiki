``ALL``
=======

Defined in header ``<Kokkos_Core.hpp>``

.. code-block:: cpp

   namespace Kokkos{
     constexpr UNSPECIFIED_TYPE ALL = IMPLEMENTATION_DETAIL;
   }

``Kokkos::ALL`` is a constant of unspecified type that is used to select all
elements in a dimension.

Example
-------

.. code-block:: cpp

   Kokkos::View<double**[5]> a("A",N0,N1);
   auto s  = Kokkos::subview(a,
                 5,
                 Kokkos::ALL,
                 Kokkos::ALL);
