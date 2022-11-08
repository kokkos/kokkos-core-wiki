``LayoutLeft``
==============

.. role:: cpp(code)
   :language: cpp

Header File: ``Kokkos_Layout.hpp``

Usage:

.. code-block:: cpp

   Kokkos::View<float*, Kokkos::LayoutLeft> my_view;

Synopsis
--------

.. code-block:: cpp

   struct LayoutLeft {

   typedef LayoutLeft array_layout;

   size_t dimension[ARRAY_LAYOUT_MAX_RANK];

   enum { is_extent_constructible = true };

   LayoutLeft(LayoutLeft const&) = default;
   LayoutLeft(LayoutLeft&&)      = default;
   LayoutLeft& operator=(LayoutLeft const&) = default;
   LayoutLeft& operator=(LayoutLeft&&) = default;

   KOKKOS_INLINE_FUNCTION
   explicit constexpr LayoutLeft(size_t N0 = 0, size_t N1 = 0, size_t N2 = 0,
                               size_t N3 = 0, size_t N4 = 0, size_t N5 = 0,
                               size_t N6 = 0, size_t N7 = 0)
     : dimension{N0, N1, N2, N3, N4, N5, N6, N7} {}
   };


Class Interface
---------------

.. cpp:class:: LayoutLeft

  This Kokkos Layout, when provided to a multidimensional View, lays out memory such that the first index is the contiguous one. This matches the Fortran conventions for allocations.

  .. rubric:: Public Member Variables

  .. cpp:member:: static constexpr bool is_extent_constructible

    A boolean enum to allow detection that this class is extent constructible

  .. cpp:member:: static constexpr unsigned dimension

    An array containing the size of each dimension of the Layout

  .. rubric:: Other Types

  .. cpp:type:: array_layout

    A tag signifying that this models the Layout concept

  .. rubric:: Constructors

  .. cpp:function:: LayoutLeft(LayoutLeft const&)

    Default copy constructor, element-wise copies the other Layout

  .. cpp:function:: LayoutLeft(LayoutLeft&&)

    Default move constructor, element-wise moves the other Layout

  .. code-block:: cpp

     KOKKOS_INLINE_FUNCTION
     explicit constexpr LayoutLeft(size_t N0 = 0, size_t N1 = 0, size_t N2 = 0,
                             size_t N3 = 0, size_t N4 = 0, size_t N5 = 0,
                             size_t N6 = 0, size_t N7 = 0);

  Constructor that takes in up to 8 sizes, to set the sizes of the corresponding dimensions of the Layout

  .. rubric:: Assignment operators

  .. cpp:function:: LayoutLeft& operator=(LayoutLeft const&) = default

    Default copy assignment, element-wise copies the other Layout

  .. cpp:function:: LayoutLeft& operator=(LayoutLeft&&) = default

    Default move assignment, element-wise moves the other Layout

