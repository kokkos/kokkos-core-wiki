# Deprecated Functionality in Kokkos 2.7

## Compile Time Detectable

 | Deprecated Feature | Replaced By | Reason for Removal |
 | --- | --- | --- | 
 | `View<>::dimension_X()` | `View<>::extent(X)` | Alignment with C++ standard |
 | `View<>::dimension(X)` | `View<>::extent(X)` | Alignment with C++ standard |
 | `View<>::capacity()` | `View<>::span(X)` | Alignment with C++ standard |
 | `View<>::operator()(Args...)` with # of Args != `View<>::rank` | `View<>::access(Args...)` | Frequent source of hard to detect bugs in user code. |
 | `View<>::View(Arg,N0,...,N7)` with # of N Args != `View<>::rank_dynamic` | Either use the constructor which uses a Layout or fix number of arguments | Frequent source of hard to detect bugs in user code. |
 | `ExecSpace::is_initialized()` | `Kokkos::is_initialized()` | Too many possibilities of organizing Kokkos initialization: now streamlined interface. |
 | `ExecSpace::initialize(...)` | `Kokkos::initialize(...)` *Note: certain overloads are gone* |  Too many possibilities of organizing Kokkos initialization: now streamlined interface. |
 | `ExecSpace::print_configuration(...)` | `Kokkos::print_configuration(...)` | Too many possibilities of organizing Kokkos initialization: now streamlined interface. |
 | `ExecSpace::max_hardware_thread_id()` | `ExecSpace::concurrency()` | Removal of execution space specific interfaces in favor of more general ones which work for all of them. | 
 | `ExecSpace::hardware_thread_id()` | Use `Kokkos::UniqueToken` | Removal of execution space specific interfaces in favor of more general ones which work for all of them. | 
 | `KOKKOS_HAVE_...` | `KOKKOS_ENABLE_...` | Harmonization of Macro Names |
 | `KOKKOS_HAVE_PTHREAD` | `KOKKOS_ENABLE_THREADS` | Harmonization of Macro Names |
 | `KOKKOS_HAVE_CXX11` | *Not necessary anymore* |
 
## Deprecated RunTime Behaviour

  | Deprecated Feature | Replaced By | Reason for Removal |
  | --- | --- | --- |
  | `deep_copy(A,B)` with A and B having not-matching dimensions | `deep_copy(subview(A,...),subview(B,...))` | Frequent source of hard to detect bugs in user code. |
  | `TeamPolicy<>(N,team_size)` with team_size larger than supported | *previously this adjusted team_size to maximum possible value, now this errors out. Use `AUTO` if team_size shall be adjustable.* | Frequent source of hard to detect bugs in user code. |