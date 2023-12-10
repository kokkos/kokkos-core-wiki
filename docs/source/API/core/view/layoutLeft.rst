``LayoutLeft``
==============

.. role:: cppkokkos(code)
   :language: cppkokkos

Header File: ``<Kokkos_Layout.hpp>``

Usage
-----

.. code-block:: cpp

   Kokkos::View<float*, Kokkos::LayoutLeft> my_view;

Description
-----------

.. cppkokkos:struct:: LayoutLeft

   When provided to a multidimensional View, lays out memory such that
   the first index is the contiguous one. This matches the Fortran conventions for allocations.

   .. rubric:: Nested Typedefs

   .. cppkokkos:type:: array_layout

       A tag signifying that this models the Layout concept.

   .. rubric:: Member Variables

   .. cppkokkos:member:: static constexpr bool is_extent_constructible = true

       A boolean to allow detection that this class is extent constructible.

   .. cppkokkos:member:: size_t dimension[8]

       An array containing the size of each dimension of the Layout.

   .. rubric:: Constructors

   .. cppkokkos:kokkosinlinefunction:: explicit constexpr LayoutLeft(size_t N0 = 0, size_t N1 = 0, \
				       size_t N2 = 0, size_t N3 = 0, size_t N4 = 0, \
				       size_t N5 = 0, size_t N6 = 0, size_t N7 = 0)

      Constructor that takes in up to 8 sizes, to set the sizes of the corresponding dimensions of the Layout.

   .. cppkokkos:function:: LayoutLeft(LayoutLeft const&) = default

       Default copy constructor, element-wise copies the other Layout.

   .. cppkokkos:function:: LayoutLeft(LayoutLeft&&) = default

       Default move constructor, element-wise moves the other Layout.

   .. rubric:: Assignment operators

   .. cppkokkos:function:: LayoutLeft& operator=(LayoutLeft const&) = default

       Default copy assignment, element-wise copies the other Layout.

   .. cppkokkos:function:: LayoutLeft& operator=(LayoutLeft&&) = default

       Default move assignment, element-wise moves the other Layout.
