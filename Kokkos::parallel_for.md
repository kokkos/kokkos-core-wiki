Defined in header `<Kokkos_Core.hpp>`

```cpp
template <class ExecPolicy, class FunctorType>
Kokkos::parallel_for(const ExecPolicy& policy, const FunctorType& functor, const std::string& str = "");
```
1. Executes `functor` in parallel according to `policy` (See [Execution Policies](API-Core#execution-policies)). The `str` name is optional, and will be used to denote this parallel region in profiling reports.
```cpp
template <class ExecPolicy, class FunctorType>
Kokkos::parallel_for(const std::string& str, const ExecPolicy& policy, const FunctorType& functor);
```
2. same as (1), the `str` name may be provided as the first argument.
```cpp
template <class FunctorType>
Kokkos::parallel_for(const size_t work_count, const FunctorType& functor, const std::string& str = "");
```
3. Equivalent to:
```cpp
Kokkos::parallel_for(Kokkos::RangePolicy<size_t>(0, work_count), functor, str);
```
See the [RangePolicy](Kokkos::RangePolicy.md) documentation.
