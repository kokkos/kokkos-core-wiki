``Kokkos::Subview``
===================

.. role:: cppkokkos(code)
   :language: cppkokkos

.. _KokkosSubview: subview.html
.. |KokkosSubview| replace:: ``Kokkos::subview``

Header File: ``Kokkos_Core.hpp``

Alias template to deduce the type that is returned by a call to the subview function with given arguments.

Usage
-----

.. code-block:: cpp

   Kokkos::Subview<ViewType,Args> subView;

Description
-----------

.. code-block:: cpp

   template <class ViewType, class... Args>
   using Subview = IMPL_DETAIL; // deduce subview type from source view traits

Type of a ``Kokkos::View`` viewing a subset of ``ViewType`` specified by ``Args...``.
Same type as returned by a call to the subview function with corresponding arguments.
For restrictions on Args see |KokkosSubview|_ documentation.

Examples
--------

.. code-block:: cpp

   using view_type = Kokkos::View<double ***[5]>;
   view_type a("A",N0,N1,N2);

   struct subViewHolder {
   Kokkos::Subview<view_type,
                   std::pair<int,int>,
                   int,
                   decltype(Kokkos::ALL),
                   int> s;
   } subViewHolder;

   subViewHolder.s  = Kokkos::subview(a,
                                      std::pair<int,int>(3,15),
                                      5,
                                      Kokkos::ALL,
                                      3);
