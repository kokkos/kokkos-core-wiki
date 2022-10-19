View and related
================

Data management is a critical part of any program. The main facility in Kokkos is the ``Kokkos::View``.
The following facilities are available:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Class
     - Description
   * - `create_mirror[_view] <view/create_mirror.html>`__
     - Creating a copy of a ``Kokkos::View`` in a different memory space.
   * - `deep_copy() <view/deep_copy.html>`__
     - Copying data between views and scalars.
   * - `LayoutLeft <view/layoutLeft.html>`__
     - Memory Layout matching Fortran.
   * - `LayoutRight <view/layoutRight.html>`__
     - Memory Layout matching C.
   * - `LayoutStride <view/layoutStride.html>`__
     - Memory Layout for arbitrary strides.
   * - `realloc <view/realloc.html>`__
     - Reallocating a ``Kokkos::View``.
   * - `resize <view/resize.html>`__
     - Resizing a ``Kokkos::View``.
   * - `subview <view/subview.html>`__
     - Getting slices from a ``Kokkos::View``.
   * - `View <view/view.html>`__
     - The main Kokkos data structure, a multidimensional memory space and layout aware array.
   * - `view_alloc() <view/view_alloc.html>`__
     - Create View allocation parameter bundle from argument list.
   * - `View-like Types <view/view_like.html>`__
     - Loosely defined as the set of class templates that behave like ``Kokkos::View``.

.. toctree::
   :hidden:
   :maxdepth: 1

   ./view/create_mirror
   ./view/deep_copy
   ./view/layoutLeft
   ./view/layoutRight
   ./view/layoutStride
   ./view/realloc
   ./view/resize
   ./view/subview
   ./view/Subview_type
   ./view/view
   ./view/view_alloc
   ./view/view_like
