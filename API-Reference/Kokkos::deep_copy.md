# Kokkos::deep_copy()

Header File: `Kokkos_Core.hpp`

Usage: 
```c++
  Kokkos::deep_copy(exec_space, dest, src);
  Kokkos::deep_copy(dest, src);
```

Copies data from `src` to `dest`, where `src` and `dest` can be [Kokkos::View](Kokkos%3A%3AView)s or scalars under certain circumstances.

## Interface

```cpp
template <class ExecSpace, class ViewDest, class ViewSrc>
void Kokkos::deep_copy(const ExecSpace& exec_space, 
                       const ViewDest& dest,
                       const ViewSrc& src);
```

```cpp
template <class ExecSpace, class ViewDest>
void Kokkos::deep_copy(const ExecSpace& exec_space, 
                       const ViewDest& dest,
                       const typename ViewDest::value_type& src);
```

```cpp
template <class ExecSpace, class ViewSrc>
void Kokkos::deep_copy(const ExecSpace& exec_space, 
                       ViewSrc::value_type& dest,
                       const ViewSrc& src);
```

```cpp
template <class ViewDest, class ViewSrc>
void Kokkos::deep_copy(const ViewDest& dest,
                       const ViewSrc& src);
```

```cpp
template <class ViewDest>
void Kokkos::deep_copy(const ViewDest& dest,
                       const typename ViewDest::value_type& src);
```

```cpp
template <class ViewSrc>
void Kokkos::deep_copy(ViewSrc::value_type& dest,
                       const ViewSrc& src);
```

### Parameters:

  * ExecSpace: An [ExecutionSpace](API-Spaces)
  * ViewDest: A [Kokkos::View](Kokkos%3A%3AView) of non-const data i.e. 
  * ViewSrc: A [Kokkos::View](Kokkos%3A%3AView).

### Requirements:
  
  * `src` is a [view-like type](ViewLike) (one of [Kokkos::View](Kokkos%3A%3AView), [Kokkos::DynRankView](Kokkos%3A%3ADynRankView), or [Kokkos::OffsetView](Kokkos%3A%3AOffsetView))
  * If `src` and `dest` are [Kokkos::View](Kokkos%3A%3AView)s, then all the following are true:
     * `std::is_same<ViewDest::non_const_value_type,ViewSrc::non_const_value_type>::value == true`
     * `src.rank ==  dest.rank`
     * For all `k` in `[0, ViewDest::rank)` `dest.extent(k) == src.extent(k)`
     * `src.span_is_contiguous() && dest.span_is_contiguous()`, OR there exists an [ExecutionSpace](API-Spaces) `copy_space` such that both `SpaceAccessibility<copy_space, ViewDest::memory_space>::accessible == true` and `SpaceAccessibility<copy_space,ViewSrc::memory_space>::accessible == true`.
  * If `src` is a [Kokkos::View](Kokkos%3A%3AView) and `dest` is a scalar, then `src.rank == 0` is true.

## Semantics

* If no [ExecutionSpace](API-Spaces) argument is provided, all outstanding operations (kernels, copy operation) in any execution spaces will be finished before the copy is executed, and the copy operation is finished before the call returns.
* If an [ExecutionSpace](API-Spaces) argument `exec_space` is provided the call is potentially asynchronous - i.e. the call returns before the copy operation is executed. In that case the copy operation will occur only after any already submitted work to `exec_space` is finished, and the copy operation will be finished before any work submitted to `exec_space` after the `deep_copy` call returns is executed. Note: the copy operation is only synchronous with respect to work in the specific execution space instance, but not necessarily with work in other instances of the same type. This behaves analogous to issuing a `cuda_memcpy_async` into a specific CUDA stream.

## Examples

```c++
#include <Kokkos_Core.hpp>
#include <cstdio>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    int N = argc > 1 ? atoi(argv[1]) : 12;
    if (N < 6) N = 12;

    // Contiguous Device View
    Kokkos::View<int**, Kokkos::LayoutLeft> d_a("A", N, 10);
    // Deep Copy Scalar into every element of a view
    Kokkos::deep_copy(d_a, 3);

    // Non Contiguous Device View
    auto d_a_2 = Kokkos::subview(d_a, 2, Kokkos::ALL);
    // Deep Copy Scalar into every element of a non-contiguous view
    Kokkos::deep_copy(d_a_2, 5);
    // Non Contiguous Device View
    auto d_a_5 = Kokkos::subview(d_a, 5, Kokkos::ALL);
    // Deep Copy between two non-contiguous views with a common execution space
    Kokkos::deep_copy(d_a_2, d_a_5);

    // Contiguous Host View
    auto h_a = Kokkos::create_mirror_view(d_a);
    // Deep Copy contiguous views
    Kokkos::deep_copy(h_a, d_a);

    // Non Contiguous Host Views
    auto h_a_2 = Kokkos::subview(h_a, 2, Kokkos::ALL);
    // Deep Copy between two non-contiguous views with potentially no common
    // execution space This fails for example if you compile the code with Cuda
    // Kokkos::deep_copy(h_a_2, d_a_2);

    // A Scalar View
    auto d_a_2_5 = Kokkos::subview(d_a, 2, 5);
    int scalar;
    // Deep Copy Scalar View into a scalar
    Kokkos::deep_copy(scalar, d_a_2_5);
  }
  Kokkos::finalize();
}
```

