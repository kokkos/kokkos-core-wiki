# Deprecation



## Kokkos-3.x


  |  **Deprecated Feature**  |  **Replacement**     | **Reason**               |
  | ------------------------ | -------------------- | ------------------------ |
  | `Kokkos::is_reducer_type` | `Kokkos::is_reducer` | Improve API
  | `OffsetView` constructors taking `index_list_type` | `Kokkos::pair` (CPU and GPU) | Streamline arguments to `::pair` function
  | overloads of `Kokkos::sort` taking a parameter `bool always_use_kokkos_sort` | Use `Kokkos::BinSort` if required, or call `Kokkos::sort` without bool parameter  | Updating overloads
  | Guard against non-public header inclusion |**Core public headers**: `Kokkos_core.hpp`, `Kokkos_Macros.hpp`, `Kokkos_Atomic.hpp`, `Kokkos_DetectionIdiom.hpp`, `Kokkos_MathematicalConstants.hpp`, `Kokkos_MathematicalFunctions.hpp`, `Kokkos_NumericTraits.hpp`, `Kokkos_Array.hpp`, `Kokkos_Complex.hpp`, `Kokkos_Pair.hpp`, `Kokkos_Half.hpp`, `Kokkos_Timer.hpp`; **Algorithms public headers**: `Kokkos_StdAlgorithms.hpp`, `Kokkos_Random.hpp`, `Kokkos_Sort.hpp`; Containers public headers: `Kokkos_Bitset.hpp`, `Kokkos_DualView.hpp`, `Kokkos_DynRankView.hpp`, `Kokkos_ErrorReporter.hpp`, `Kokkos_Functional.hpp`, `Kokkos_OffsetView.hpp`, `Kokkos_ScatterView.hpp`, `Kokkos_StaticCrsGraph.hpp`, `Kokkos_UnorderedMap.hpp`, `Kokkos_Vector.hpp` | Improve API
  | Raise deprecation warnings if non-empty WorkTag class is used | Use empty WorkTag class | Improve API
  | `parallel_*` overloads taking the label as trailing argument | `Kokkos::parallel_*("KokkosViewLabel", policy, f);` | Consistent ordering of parameters
  | Embedded types (`argument_type`, `first_argument_type`, and `second_argument_type`) in `std::function` | Use `decltype` if required | Deprecation in `Kokkos_Functional.hpp` mirrors that in `std::function` (`#include <functional>`) 
  | `InitArguments` struct | `InitializationSettings()` object with attributes that can be queried | Make initialization transparent and understandable
  | `finalize_all()` | `finalize()`| Improve  API
  | Command-line arguments (other than `--help`) that are not prefixed with `kokkos-*` | `--kokkos-num-threads`, `--kokkos-device-id`, `--kokkos-num-devices`| Improve API
  | `--[kokkos-]numa` command-line argument and `KOKKOS_NUMA` environment variable | `--kokkos-num-threads`| Harmonize option nomenclature with that of C++ `std::thread` library
  | `--[kokkos-]threads` command-line argument | `--kokkos-num-threads` | Improve API
  | Warn about `parallel_reduce` cases that call `join()` with arguments qualified by `volatile` keyword | Remove `volatile` overloads | Streamline API
  | `static void partition_master(F const& f, int requested_num_partitions = 0, int requested_partition_size = 0)` | Remove function | Improve API
  | `void OpenMPInternal::validate_partition_impl(const int nthreads, int &num_partitions, int &partition_size)` | Remove function | Improve API
  | `std::vector<OpenMP> OpenMP::partition(...) { return std::vector<OpenMP>(1); }` | Remove function | Improve API
  | `OpenMP OpenMP::create_instance(...) { return OpenMP(); }` | Remove function | Improve API
  | `static void validate_partition(const int nthreads, int& num_partitions, int& partition_size)` | Remove function | Improve API
  | `static void validate_partition_impl(const int nthreads, int& num_partitions, int& partition_size)` | Remove function | Improve API
  | `void OpenMP::partition_master(F const& f, int num_partitions, int partition_size) | Remove function | Improve API
  | `class MasterLock<OpenMP>` | Remove class | Improve API  

