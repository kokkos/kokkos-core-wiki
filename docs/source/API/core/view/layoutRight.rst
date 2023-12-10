``LayoutRight``
===============

.. role:: cppkokkos(code)
   :language: cppkokkos

Header File: ``<Kokkos_Layout.hpp>``

Usage
-----

.. code-block:: cpp

   Kokkos::View<float*, Kokkos::LayoutRight> my_view;

Description
-----------

.. cppkokkos:struct:: LayoutRight

   When provided to a multidimensional View, lays out memory such that the
   **last index is the contiguous one**. This matches the C conventions for allocations.

   .. rubric:: Nested Typedefs

   .. cppkokkos:type:: array_layout

       A tag signifying that this models the Layout concept.

   .. rubric:: Member Variables

   .. cppkokkos:member:: static constexpr bool is_extent_constructible = true

       A boolean to allow detection that this class is extent constructible.

   .. cppkokkos:member:: size_t dimension[8]

       An array containing the size of each dimension of the Layout.

   .. rubric:: Constructors

   .. cppkokkos:kokkosinlinefunction:: explicit constexpr LayoutRight(size_t N0 = 0, size_t N1 = 0, \
				       size_t N2 = 0, size_t N3 = 0, size_t N4 = 0, \
				       size_t N5 = 0, size_t N6 = 0, size_t N7 = 0)

      Constructor that takes in up to 8 sizes, to set the sizes of the corresponding dimensions of the Layout.

   .. cppkokkos:function:: LayoutRight(LayoutRight const&) = default

       Default copy constructor, element-wise copies the other Layout.

   .. cppkokkos:function:: LayoutRight(LayoutRight&&) = default

       Default move constructor, element-wise moves the other Layout.

   .. rubric:: Assignment operators

   .. cppkokkos:function:: LayoutRight& operator=(LayoutRight const&) = default

       Default copy assignment, element-wise copies the other Layout.

   .. cppkokkos:function:: LayoutRight& operator=(LayoutRight&&) = default

       Default move assignment, element-wise moves the other Layout.
