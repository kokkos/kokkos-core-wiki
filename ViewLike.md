View-like Types
===============

View-like types are loosely defined as the set of class templates that behave like `Kokkos::View` from an interface perspective. There is not a full formal definition of what this means yet, which means there is no way for users to add to this list in a way that the new class is recognized by Kokkos facilities operating on View-like things.  In Kokkos these class templates are considered View-like: 
  * [Kokkos::View](Kokkos%3A%3AView) 
  * [Kokkos::DynRankView](Kokkos%3A%3ADynRankView)
  * [Kokkos::OffsetView](Kokkos%3A%3AOffsetView)
  * [Kokkos::DynamicView](Kokkos%3A%3ADynamicView)

Notably, [Kokkos::DualView](Kokkos%3A%3ADualView) and [Kokkos::ScatterView](Kokkos%3A%3AScatterView) are **not** included in this category. 