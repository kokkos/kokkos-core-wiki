# Deprecation in Kokkos 3.7.00



## Compile Time Detectable

 | Deprecated Feature | Replaced By | Reason for Removal |
 | ------------------ | ------------| ------------------ | 
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
 | `ExecSpace::fence()` | Use `ExecSpace().fence()` | Support for instances, where you want to only fence that instance need this to be a non-static member function |
 | `ExecSpace::is_initialized()` | Use `Kokkos::is_initialized()` | Simplify initialization makes this superfluous |
 | `TeamPolicy<>::team_size_max(Functor)` | `TeamPolicy<>::team_size_max(Functor, DispatchTag)` | This is now a member function of the team policy. The previous variant didn't take all necessary information into account and could result in invalid answers. |
 | `TeamPolicy<>::team_size_recommended(Functor)` | `TeamPolicy<>::team_size_recommended(Functor, DispatchTag)` | This is now a member function of the team policy. The previous variant didn't take all necessary information into account and could result in invalid answers. |
 | `KOKKOS_HAVE_...` | `KOKKOS_ENABLE_...` | Harmonization of Macro Names |
 | `KOKKOS_HAVE_PTHREAD` | `KOKKOS_ENABLE_THREADS` | Harmonization of Macro Names |
 | `KOKKOS_HAVE_CXX11` | *Not necessary anymore* |
 | `DualView::modified_host` | The modify flags are now private members and not accessible. Use `DualView::clear_sync_state()` to reset the modification markers | This allowed us internal optimization such as having both views being merged into one and deciding where to store the data. |
 | `DualView::modified_device` | The modify flags are now private members and not accessible. Use `DualView::clear_sync_state()` to reset the modification markers | This allowed us internal optimization such as having both views being merged into one and deciding where to store the data. |
 
## Deprecated RunTime Behaviour

  | Deprecated Feature | Replaced By | Reason for Removal |
  | ------------------ | ------------| ------------------ | 
  | `deep_copy(A,B)` with A and B having not-matching dimensions | `deep_copy(subview(A,...),subview(B,...))` | Frequent source of hard to detect bugs in user code. |
  | `TeamPolicy<>(N,team_size)` with team_size larger than supported | *previously this adjusted team_size to maximum possible value, now this errors out. Use `AUTO` if team_size shall be adjustable.* | Frequent source of hard to detect bugs in user code. |
  | `DualView::sync<SPACE>` with `SPACE` not matching the actual underlying views Spaces | Call with the right `SPACE` type or explicitly call sync_host(), sync_device() if templating is not needed. | This had unexpected behaviour in particular when using DualViews on UVM memory. Furthermore every non-matching space would end up synching the host. |
  | `DualView::need_sync<SPACE>` with `SPACE` not matching the actual underlying views Spaces | Call with the right `SPACE` type or explicitly call need_sync_host(), need_sync_device() if templating is not needed. | This had unexpected behaviour in particular when using DualViews on UVM memory. Furthermore every non-matching space would behave as if asked for state of the host view. |
  | `DualView::modify<SPACE>` with `SPACE` not matching the actual underlying views Spaces | Call with the right `SPACE` type or explicitly call modify_host(), modify_device() if templating is not needed. | This had unexpected behaviour in particular when using DualViews on UVM memory. Furthermore every non-matching space would end up marking the host view as modified. |
  | `DualView::view<SPACE>` with `SPACE` not matching the actual underlying views Spaces | Call with the right `SPACE` type or explicitly call view_host(), view_device() if templating is not needed. | This had unexpected behaviour in particular when using DualViews on UVM memory. Furthermore every non-matching space would end up giving back the host.
  | `Kokkos::is_reducer_type` | `Kokkos::is_reducer` | Streamline API semantics
  | `OffsetView` constructors taking `index_list_type` | `std::pair (CPU), Kokkos::pair (GPU)` | Streamline arguments to `::pair` function
  | overloads of `Kokkos::sort` taking a parameter `bool always_use_kokkos_sort` | Remove `always_use_kokkos_sort` bool parameter  | Updating overloads
  | Guard against non-public header inclusion | Core public headers: `Kokkos_core.cpp`, `Kokkos_Macros.hpp`, `Kokkos_Atomic.hpp`, `Kokkos_DetectionIdiom`, `Kokkos_MathematicalConstants.hpp`, `Kokkos_MathematicalFunctions.hpp`, `Kokkos_NumericTraits.hpp`, `Kokkos_Array.hpp`, `Kokkos_Complex.hpp`, `Kokkos_Pair.hpp`, `Kokkos_Half.hpp`, `Kokkos_Timer.hpp`; Algorithms public headers: `Kokkos_StdAlgorithms.hpp`, `Kokkos_Random.hpp`, `Kokkos_Sort.hpp`; Containers public headers: `Kokkos_Bit.hpp`, `Kokkos_DualView.hpp`, `Kokkos_DynRankView.hpp`, `Kokkos_ErrorReporter.hpp`, `Kokkos_Functional.hpp`, `Kokkos_OffsetView.hpp`, `Kokkos_ScatterView.hpp`, `Kokkos_StaticCrsGraph.hpp`, `Kokkos_UnorderedMap.hpp`, `Kokkos_Vector.hpp` | Improve API and reduce build time
  | Raise deprecation warnings if non-empty WorkTag class is used | in Kokkos-4.0, use empty WorkTag class | Improve API
  | `parallel_*` overloads taking the label as trailing argument | `Kokkos::parallel_*("KokkosViewLabel", policy, f);` | More logical ordering of parameters
  | embedded types (`argument_type`, `first_argument_type`, and `second_argument_type`) in `std::function` | types removed in C++20 | Deprecation in `Kokkos_Functional.hpp` mirrors that in `std::function` (`#include <functional>`) 
  | `InitArguments` struct | `InitializationSettings()` object with attributes that can be queried | Align with object-oriented programming
  | `finalize_all()` | `finalize_spaces()`| Improve  API
  | command line arguments (other than `--help`) that are not prefixed with `kokkos-*` | `--kokkos-num-threads`, `--kokkos-device-id`, `--kokkos-num-devices`| Improve API
  | `--[kokkos-]numa` command line arg and `KOKKOS_NUMA` env var | `--kokkos-num-threads`| Harmonize option name with that of C++ `std::thread` library
  | `--[kokkos-]threads` command line argument in favor of `--[kokkos-]num-threads` | `--kokkos-num-threads` | Improve API 
  | Warn about `parallel_reduce` cases that call `join()` with arguments qualified by `volatile` keyword | `volatile` will be removed in Kokkos-4.0 | Track C++ Standard evolution
