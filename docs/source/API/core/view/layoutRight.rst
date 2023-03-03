``LayoutRight``
===============

.. role:: cppkokkos(code)
    :language: cppkokkos

Header File: ``<Kokkos_Layout.hpp>``

Usage
-----

.. code-block:: cpp

    Kokkos::View<float*, Kokkos::LayoutRight> my_view;

Class Interface
---------------

.. cppkokkos:struct:: LayoutRight

    This Kokkos Layout, when provided to a multidimensional View, lays out memory such that the last index is the contiguous one. This matches the C conventions for allocations.

    .. rubric:: Public Member Variables

    .. cppkokkos:member:: static constexpr bool is_extent_constructible

        A boolean enum to allow detection that this class is extent constructible.

    .. cppkokkos:member:: static constexpr unsigned dimension

        An array containing the size of each dimension of the Layout.

    .. rubric:: Other Types

    .. cppkokkos:type:: array_layout

        A tag signifying that this models the Layout concept.

    .. rubric:: Constructors

    .. cppkokkos:function:: LayoutRight(LayoutRight const&);

        Default copy constructor, element-wise copies the other Layout.

    .. cppkokkos:function:: LayoutRight(LayoutRight&&);

        Default move constructor, element-wise moves the other Layout.

    .. cppkokkos:function:: KOKKOS_INLINE_FUNCTION explicit constexpr LayoutRight(size_t N0 = 0, size_t N1 = 0, size_t N2 = 0, size_t N3 = 0, size_t N4 = 0, size_t N5 = 0, size_t N6 = 0, size_t N7 = 0);

        Constructor that takes in up to 8 sizes, to set the sizes of the corresponding dimensions of the Layout.

    .. rubric:: Assignment operators

    .. cppkokkos:function:: LayoutRight& operator=(LayoutRight const&) = default;

        Default copy assignment, element-wise copies the other Layout.

    .. cppkokkos:function:: LayoutRight& operator=(LayoutRight&&) = default;

        Default move assignment, element-wise moves the other Layout.
