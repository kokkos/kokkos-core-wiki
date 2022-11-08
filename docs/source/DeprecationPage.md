# Deprecation


## Kokkos-4.0 and Kokkos-3.7.0


  | Deprecated Feature  | Replaced By          | Reason for Removal   |
  | --------------------| -------------------- | -------------------- |
  | `Kokkos::is_reducer_type` | `Kokkos::is_reducer` | Streamline API
  | `OffsetView` constructors taking `index_list_type` | `std::pair (CPU), Kokkos::pair (GPU)` | Streamline arguments to `::pair` function
  | overloads of `Kokkos::sort` taking a parameter `bool always_use_kokkos_sort` | Remove `always_use_kokkos_sort` bool parameter  | Updating overloads
  | Guard against non-public header inclusion | Core public headers: `Kokkos_core.cpp`, `Kokkos_Macros.hpp`, `Kokkos_Atomic.hpp`, `Kokkos_DetectionIdiom`, `Kokkos_MathematicalConstants.hpp`, `Kokkos_MathematicalFunctions.hpp`, `Kokkos_NumericTraits.hpp`, `Kokkos_Array.hpp`, `Kokkos_Complex.hpp`, `Kokkos_Pair.hpp`, `Kokkos_Half.hpp`, `Kokkos_Timer.hpp`; Algorithms public headers: `Kokkos_StdAlgorithms.hpp`, `Kokkos_Random.hpp`, `Kokkos_Sort.hpp`; Containers public headers: `Kokkos_Bit.hpp`, `Kokkos_DualView.hpp`, `Kokkos_DynRankView.hpp`, `Kokkos_ErrorReporter.hpp`, `Kokkos_Functional.hpp`, `Kokkos_OffsetView.hpp`, `Kokkos_ScatterView.hpp`, `Kokkos_StaticCrsGraph.hpp`, `Kokkos_UnorderedMap.hpp`, `Kokkos_Vector.hpp` | Improve API and reduce build time
  | Raise deprecation warnings if non-empty WorkTag class is used | Use empty WorkTag class | Improve API
  | `parallel_*` overloads taking the label as trailing argument | `Kokkos::parallel_*("KokkosViewLabel", policy, f);` | More logical ordering of parameters
  | Embedded types (`argument_type`, `first_argument_type`, and `second_argument_type`) in `std::function` | types removed in C++20 | Deprecation in `Kokkos_Functional.hpp` mirrors that in `std::function` (`#include <functional>`) 
  | `InitArguments` struct | `InitializationSettings()` object with attributes that can be queried | Align with object-oriented programming
  | `finalize_all()` | `finalize_spaces()`| Improve  API
  | Command-line arguments (other than `--help`) that are not prefixed with `kokkos-*` | `--kokkos-num-threads`, `--kokkos-device-id`, `--kokkos-num-devices`| Improve API
  | `--[kokkos-]numa` command-line argument and `KOKKOS_NUMA` environment variable | `--kokkos-num-threads`| Harmonize option nomenclature with that of C++ `std::thread` library
  | `--[kokkos-]threads` command-line argument | `--kokkos-num-threads` | Improve API
  | Warn about `parallel_reduce` cases that call `join()` with arguments qualified by `volatile` keyword | `volatile` will be removed | Track C++ Standard evolution
