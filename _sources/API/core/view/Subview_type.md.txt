# `Kokkos::Subview`

Header File: `Kokkos_Core.hpp`

Alias template to deduce the type that is returned by a call to the subview function with given arguments.

Usage:
```c++
Kokkos::Subview<ViewType,Args> subView;
```

## Description

```c++
template <class ViewType, class... Args>
using Subview = IMPL_DETAIL; // deduce subview type from source view traits
```
Type of a `Kokkos::View` viewing a subset of `ViewType` specified by `Args...`.
Same type as returned by a call to the subview function with corresponding arguments.
For restrictions on Args see [`Kokkos::subview`](Kokkos%3A%3Asubview) documentation.
   
## Examples

```c++

using view_type = Kokkos::View<double ***[5]>;
view_type a("A",N0,N1,N2);

struct subViewHolder {
Kokkos::Subview<view_type,
                std::pair<int,int>,
                int,
                Kokkos::ALL,int> s;
} subViewHolder;

subViewHolder.s  = Kokkos::subview(a,
                                   std::pair<int,int>(3,15),
                                   5,
                                   Kokkos::ALL,
                                   3);

```
