
## Kokkos::Experimental::{begin, cbegin, end, cend}

Header File: `Kokkos_Core.hpp`

```cpp
template <class DataType, class... Properties>
KOKKOS_INLINE_FUNCTION auto begin(const Kokkos::View<DataType, Properties...>& view);  (1)

template <class DataType, class... Properties>
KOKKOS_INLINE_FUNCTION auto cbegin(const Kokkos::View<DataType, Properties...>& view); (2)

template <class DataType, class... Properties>
KOKKOS_INLINE_FUNCTION auto end(const Kokkos::View<DataType, Properties...>& view);    (3)

template <class DataType, class... Properties>
KOKKOS_INLINE_FUNCTION auto cend(const Kokkos::View<DataType, Properties...>& view);   (4)
```

### Description

- (1,2) return a Kokkos iterator to the beginning of `view`
- (3,4) return a Kokkos iterator to the element past the end of `view`

### Notes
- for performance reasons, these functions currently return a **random access** iterator

- `cbegin, cend` ensure that the dereferenced iterator is const-qualified.

- `view` is taken as `const` because, within each function, we are not changing the view itself: the returned iterator operates on the view without changing its structure.

- dereferencing an iterator must be done within an execution space where `view` is accessible

### Parameters and Requirements

- `view`: must be a rank-1 view with `LayoutLeft`, `LayoutRight`, or `LayoutStride`

### Example
```cpp
namespace KE = Kokkos::Experimental;
using view_type = Kokkos::View<int*>;
view_type a("a", 15);

auto it = KE::begin(a);
// if dereferenced (within a proper execution space), can modify the content of `a`

auto itc = KE::cbegin(a);
// if dereferenced (within a proper execution space), can only read the content of `a`
```

