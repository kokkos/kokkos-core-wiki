
# `find`

Header File: `Kokkos_StdAlgorithms.hpp`

```cpp
namespace Kokkos{
namespace Experimental{

template <class ExecutionSpace, class InputIterator, class T>
InputIterator find(const ExecutionSpace& exespace,                                   (1)
                   InputIterator first, InputIterator last,
                   const T& value);

template <class ExecutionSpace, class InputIterator, class T>
InputIterator find(const std::string& label, const ExecutionSpace& exespace,         (2)
                   InputIterator first, InputIterator last,
                   const T& value);

template <class ExecutionSpace, class DataType, class... Properties, class T>
auto find(const ExecutionSpace& exespace,                                            (3)
          const ::Kokkos::View<DataType, Properties...>& view,
          const T& value);

template <class ExecutionSpace, class DataType, class... Properties, class T>
auto find(const std::string& label, const ExecutionSpace& exespace,                  (4)
          const ::Kokkos::View<DataType, Properties...>& view,
          const T& value);

} //end namespace Experimental
} //end namespace Kokkos
```

## Description

Returns an iterator to the *first* element in `[first, last)` (1,2) or ``view`` (3,4) 
that equals `value`. Equality is checked using `operator==`.

## Parameters and Requirements

- `exespace`:
  - execution space instance

- `label`:
  - string forwarded to all implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::find_iterator_api_default"
  - for 3, the default string is: "Kokkos::find_view_api_default"

- `first, last`:
  - range of elements to search in
  - must be *random access iterators*, e.g., returned from `Kokkos::Experimental::(c)begin/(c)end`
  - must represent a valid range, i.e., `last >= first` (this condition is checked in debug mode)
  - must be accessible from `exespace`

- `view`:
  - must be rank-1, and have `LayoutLeft`, `LayoutRight`, or `LayoutStride`
  - must be accessible from `exespace`

## Return

- (1,2): `InputIterator` instance pointing to the first element that equals `value`, or `last` if no elements is found
- (2,3): iterator to the first element that equals `value`, or `Kokkos::Experimental::end(view)` if none is found

## Example

```cpp
namespace KE = Kokkos::Experimental;
auto exespace = Kokkos::DefaultExecutionSpace;
using view_type = Kokkos::View<exespace, int*>;
view_type a("a", 15);
// fill "a" somehow

auto exespace = Kokkos::DefaultExecutionSpace;
auto it1 = KE::find(exespace, KE::cbegin(a), KE::cend(a), 5);

// assuming OpenMP is enabled and "a" is host-accessible, you can also do
auto it2 = KE::find(Kokkos::OpenMP(), KE::begin(a), KE::end(a), 5);
```
