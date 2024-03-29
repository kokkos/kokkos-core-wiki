
# `generate`

Header File: `Kokkos_StdAlgorithms.hpp`

```c++
namespace Kokkos{
namespace Experimental{

template <class ExecutionSpace, class IteratorType, class GeneratorType>
void generate(const ExecutionSpace& exespace,                                (1)
              IteratorType first, IteratorType last,
              GeneratorType g);

template <class ExecutionSpace, class IteratorType, class GeneratorType>
void generate(const std::string& label, const ExecutionSpace& exespace,      (2)
              IteratorType first, IteratorType last,
              GeneratorType g);

template <class ExecutionSpace, class DataType, class... Properties, class GeneratorType>
void generate(const ExecutionSpace& exespace,                                (3)
              const Kokkos::View<DataType, Properties...>& view,
              GeneratorType g);

template <class ExecutionSpace, class DataType, class... Properties, class GeneratorType>
void generate(const std::string& label, const ExecutionSpace& exespace,      (4)
              const Kokkos::View<DataType, Properties...>& view,
              GeneratorType g);

} //end namespace Experimental
} //end namespace Kokkos
```

## Description

Assigns the value generated by the functor `g` to each elements in the
range `[first, last)` (overloads 1,2) or in the `view` (overloads 3,4).


## Parameters and Requirements

- `exespace`:
  - execution space instance
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::generate_iterator_api_default"
  - for 3, the default string is: "Kokkos::generate_view_api_default"
- `first, last`:
  - range of elements to modify
  - must be *random access iterators*
  - must represent a valid range, i.e., `last >= first` (checked in debug mode)
  - must be accessible from `exespace`
- `view`:
  - view to modify
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`
- `g`:
  - functor of the form:
  ```c++
  struct Generate
  {
      KOKKOS_INLINE_FUNCTION
      return_type operator()() const{ return /* ... */; }
  };
  ```
  where `return_type` must be assignable to `value_type`, with `value_type`
  being the value type of `IteratorType` (for 1,2) or of `view` (for 3,4).


## Return

None
