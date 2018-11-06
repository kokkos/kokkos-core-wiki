# Deprecated Functionality in Kokkos 2.7.24

## Compile Time Detectable

 | Deprecated Feature | Replaced By | Reason for Removal |
 | --- | --- | --- | 
 | `View<>::ptr_on_device()` | `View<>::data()` | Alignment with C++ standard |
 | `View<>::dimension_X()` | `View<>::extent(X)` | Alignment with C++ standard |
 | `View<>::dimension(X)` | `View<>::extent(X)` | Alignment with C++ standard |
 | `View<>::capacity()` | `View<>::span(X)` | Alignment with C++ standard |
 | `View<>::operator()(Args...)` with # of Args != `View<>::rank` | `View<>::access(Args...)` | Frequent source of hard to detect bugs in user code. |
 | `View<>::View(Arg,N0,...,N7)` with # of N Args != `View<>::rank_dynamic` | Either fix number of arguments, or create a meta function to produce layout with correct number of arguments. Note you can also use the implementation detail macro KOKKOS_IMPL_CTOR_DEFAULT_ARG for arguments larger rank_dynamic. But there is no backwards compatibility guarantee on that. | Frequent source of hard to detect bugs in user code. |
 | `ExecSpace::is_initialized()` | `Kokkos::is_initialized()` | Too many possibilities of organizing Kokkos initialization: now streamlined interface. |
 | `ExecSpace::initialize(...)` | `Kokkos::initialize(...)` *Note: certain overloads are gone* |  Too many possibilities of organizing Kokkos initialization: now streamlined interface. |
 | `ExecSpace::print_configuration(...)` | `Kokkos::print_configuration(...)` | Too many possibilities of organizing Kokkos initialization: now streamlined interface. |
 | `ExecSpace::max_hardware_thread_id()` | `ExecSpace::concurrency()` | Removal of execution space specific interfaces in favor of more general ones which work for all of them. | 
 | `ExecSpace::hardware_thread_id()` | Use `Kokkos::UniqueToken` | Removal of execution space specific interfaces in favor of more general ones which work for all of them. | 
 | `TeamPolicy<>::team_size_max(Functor)` | `TeamPolicy<>::team_size_max(Functor, DispatchTag)` | This is now a member function of the team policy. The previous variant didn't take all necessary information into account and could result in invalid answers. |
 | `TeamPolicy<>::team_size_recommended(Functor)` | `TeamPolicy<>::team_size_recommended(Functor, DispatchTag)` | This is now a member function of the team policy. The previous variant didn't take all necessary information into account and could result in invalid answers. |
 | `KOKKOS_HAVE_...` | `KOKKOS_ENABLE_...` | Harmonization of Macro Names |
 | `KOKKOS_HAVE_PTHREAD` | `KOKKOS_ENABLE_THREADS` | Harmonization of Macro Names |
 | `KOKKOS_HAVE_CXX11` | *Not necessary anymore* |
 | `DualView::modified_host` | The modify flags are now private members and not accessible. Use `DualView::clear_sync_state()` to reset the modification markers | This allowed us internal optimization such as having both views being merged into one and deciding where to store the data. |
 | `DualView::modified_device` | The modify flags are now private members and not accessible. Use `DualView::clear_sync_state()` to reset the modification markers | This allowed us internal optimization such as having both views being merged into one and deciding where to store the data. |
 
## Deprecated RunTime Behaviour

  | Deprecated Feature | Replaced By | Reason for Removal |
  | --- | --- | --- |
  | `deep_copy(A,B)` with A and B having not-matching dimensions | `deep_copy(subview(A,...),subview(B,...))` | Frequent source of hard to detect bugs in user code. |
  | `TeamPolicy<>(N,team_size)` with team_size larger than supported | *previously this adjusted team_size to maximum possible value, now this errors out. Use `AUTO` if team_size shall be adjustable.* | Frequent source of hard to detect bugs in user code. |
  | `DualView::sync<SPACE>` with `SPACE` not matching the actual underlying views Spaces | Call with the right `SPACE` type or explicitly call sync_host(), sync_device() if templating is not needed. | This had unexpected behaviour in particular when using DualViews on UVM memory. Furthermore every non-matching space would end up synching the host. |
  | `DualView::need_sync<SPACE>` with `SPACE` not matching the actual underlying views Spaces | Call with the right `SPACE` type or explicitly call need_sync_host(), need_sync_device() if templating is not needed. | This had unexpected behaviour in particular when using DualViews on UVM memory. Furthermore every non-matching space would behave as if asked for state of the host view. |
  | `DualView::modify<SPACE>` with `SPACE` not matching the actual underlying views Spaces | Call with the right `SPACE` type or explicitly call modify_host(), modify_device() if templating is not needed. | This had unexpected behaviour in particular when using DualViews on UVM memory. Furthermore every non-matching space would end up marking the host view as modified. |
  | `DualView::view<SPACE>` with `SPACE` not matching the actual underlying views Spaces | Call with the right `SPACE` type or explicitly call view_host(), view_device() if templating is not needed. | This had unexpected behaviour in particular when using DualViews on UVM memory. Furthermore every non-matching space would end up giving back the host. |
