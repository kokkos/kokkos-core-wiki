(KokkosALL)=
# `Kokkos::ALL`

Defined in header `<Kokkos_Core.hpp>`

```c++
namespace Kokkos{
  KokkosInternalType ALL = impl_detail;
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
