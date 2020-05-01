View-like Types
===============

View-like types are loosely defined as the set of class templates that behave like `Kokkos::View` from an interface perspective. There is not a full formal definition of what this means yet, but in Kokkos these class templates include `Kokkos::View`, `Kokkos::DynRankView`, `Kokkos::OffsetView`, and `Kokkos::DynamicView`. Notably, `Kokkos::DualView` is **not** included in this category. 