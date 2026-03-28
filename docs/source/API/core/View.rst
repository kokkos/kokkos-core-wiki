View and related
================

Data management is a critical part of any program. The main facility in Kokkos is the ``Kokkos::View``.
The following facilities are available:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Class
     - Description
   * - :doc:`create_mirror[_view] <view/create_mirror>`
     - Creating a copy of a ``Kokkos::View`` in a different memory space.
   * - :doc:`deep_copy() <view/deep_copy>`
     - Copying data between views and scalars.
   * - :doc:`LayoutLeft <view/layoutLeft>`
     - Memory Layout matching Fortran.
   * - :doc:`LayoutRight <view/layoutRight>`
     - Memory Layout matching C.
   * - :doc:`LayoutStride <view/layoutStride>`
     - Memory Layout for arbitrary strides.
   * - :doc:`MemoryTraits <view/memoryTraits>`
     - Memory access traits.
   * - :doc:`realloc <view/realloc>`
     - Reallocating a ``Kokkos::View``.
   * - :doc:`resize <view/resize>`
     - Resizing a ``Kokkos::View``.
   * - :doc:`subview <view/subview>`
     - Getting slices from a ``Kokkos::View``.
   * - :doc:`View <view/view>`
     - The main Kokkos data structure, a multidimensional memory space and layout aware array.
   * - :doc:`view_alloc() <view/view_alloc>`
     - Create View allocation parameter bundle from argument list.
   * - :doc:`View-like Types <view/view_like>`
     - Loosely defined as the set of class templates that behave like ``Kokkos::View``.

.. toctree::
   :hidden:
   :maxdepth: 1

   ./view/create_mirror
   ./view/deep_copy
   ./view/layoutLeft
   ./view/layoutRight
   ./view/layoutStride
   ./view/memoryTraits
   ./view/realloc
   ./view/resize
   ./view/subview
   ./view/Subview_type
   ./view/view
   ./view/view_alloc
   ./view/view_like
