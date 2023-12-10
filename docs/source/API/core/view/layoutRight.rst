``LayoutRight``
===============

.. role:: cpp(code)
   :language: cpp

Header File: ``Kokkos_Layout.hpp``

Usage: 

.. code-block:: cpp

   Kokkos::View<float*, Kokkos::LayoutRight> my_view;

Synopsis
--------

.. code-block:: c++

   struct LayoutRight {

   typedef LayoutRight array_layout;

   size_t dimension[ARRAY_LAYOUT_MAX_RANK];

   enum { is_extent_constructible = true };

   LayoutRight(LayoutRight const&) = default;
   LayoutRight(LayoutRight&&)      = default;
   LayoutRight& operator=(LayoutRight const&) = default;
   LayoutRight& operator=(LayoutRight&&) = default;

   KOKKOS_INLINE_FUNCTION
   explicit constexpr LayoutRight(size_t N0 = 0, size_t N1 = 0, size_t N2 = 0,
                               size_t N3 = 0, size_t N4 = 0, size_t N5 = 0,
                               size_t N6 = 0, size_t N7 = 0)
     : dimension{N0, N1, N2, N3, N4, N5, N6, N7} {}
   };

Class Interface
---------------

.. cpp:class:: LayoutRight

  This Kokkos Layout, when provided to a multidimensional View, lays out memory such that the last index is the contiguous one. This matches the C conventions for allocations.

  .. rubric:: Public Member Variables

  .. cpp:member:: static constexpr bool is_extent_constructible

    A boolean enum to allow detection that this class is extent constructible

  .. cpp:member:: static constexpr unsigned dimension

    An array containing the size of each dimension of the Layout

  .. rubric:: Other Types

  .. cpp:type:: array_layout

    A tag signifying that this models the Layout concept

  .. rubric:: Constructors

  .. cpp:function:: LayoutRight(LayoutRight const&)

    Default copy constructor, element-wise copies the other Layout

  .. cpp:function:: LayoutRight(LayoutRight&&)

    Default move constructor, element-wise moves the other Layout

  .. code-block:: cpp

     KOKKOS_INLINE_FUNCTION
     explicit constexpr LayoutRight(size_t N0 = 0, size_t N1 = 0, size_t N2 = 0,
                             size_t N3 = 0, size_t N4 = 0, size_t N5 = 0,
                             size_t N6 = 0, size_t N7 = 0);

  Constructor that takes in up to 8 sizes, to set the sizes of the corresponding dimensions of the Layout

  .. rubric:: Assignment operators

  .. cpp:function:: LayoutRight& operator=(LayoutRight const&) = default

    Default copy assignment, element-wise copies the other Layout

  .. cpp:function:: LayoutRight& operator=(LayoutRight&&) = default

    Default move assignment, element-wise moves the other Layout

