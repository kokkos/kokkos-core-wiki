View-like Types
===============

View-like types are loosely defined as the set of class templates that behave like `Kokkos::View <view.html>`__ from an interface perspective. There is not a full formal definition of what this means yet, which means there is no way for users to add to this list in a way that the new class is recognized by Kokkos facilities operating on View-like things. In Kokkos these class templates are considered View-like:

* `Kokkos::View <view.html>`_
* `Kokkos::DynRankView <../../containers/DynRankView.html>`_
* `Kokkos::OffsetView <../../containers/Offset-View.html>`_
* `Kokkos::DynamicView <../../containers/DynamicView.html>`_

Notably, `Kokkos::DualView <../../containers/DualView.html>`_ and `Kokkos::ScatterView <../../containers/ScatterView.html>`_ are **not** included in this category.
