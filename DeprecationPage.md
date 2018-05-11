# Deprecated Functionality in Kokkos 2.7

## Compile Time Detectable

 | Deprecated Feature | Replaced By | 
 | --- | --- | 
 | `View<>::dimension_X()` | `View<>::extent(X)` |
 | `View<>::dimension(X)` | `View<>::extent(X)` |
 | `View<>::capacity()` | `View<>::span(X)` |
 | `View<>::operator()(Args...)` with # of Args != `View<>::rank` | `View<>::access(Args...)` |
 | `ExecSpace::is_initialized()` | `Kokkos::is_initialized()` |
 | `ExecSpace::initialize(...)` | `Kokkos::initialize(...)` *Note: certain overloads are gone* |
 | `ExecSpace::max_hardware_thread_id()` | `ExecSpace::concurrency()` |
 | `ExecSpace::hardware_thread_id()` | Use `Kokkos::UniqueToken` |
 | `KOKKOS_HAVE_...` | `KOKKOS_ENABLE_...` |
 | `KOKKOS_HAVE_PTHREAD` | `KOKKOS_ENABLE_THREADS` |
 | `KOKKOS_HAVE_CXX11` | *Not necessary anymore* |
 
## Deprecated RunTime Behaviour

  | Deprecated Feature | Replaced By |
  | --- | --- | 
  | `deep_copy(A,B)` with A and B having not-matching dimensions | `deep_copy(subview(A,...),subview(B,...))` |
  | `TeamPolicy<>(N,team_size)` with team_size larger than supported | *previously this adjusted team_size to maximum possible value, now this errors out. Use `AUTO` if team_size shall be adjustable.* |