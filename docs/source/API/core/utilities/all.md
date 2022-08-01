(KokkosALL)=
# `Kokkos::ALL`

Defined in header `<Kokkos_Core.hpp>`

```c++
namespace Kokkos{
  constexpr UNSPECIFIED-TYPE ALL = IMPLEMENTATION-DETAIL;
}
```

Type used as tag to specify the selection of all elements in a dimension.


## Examples

```c++
Kokkos::View<double***[5]> a("A",N0,N1,N2);

auto s  = Kokkos::subview(a,
              std::pair<int,int>(3,15),
              5,
              Kokkos::ALL,
              Kokkos::ALL);
```
