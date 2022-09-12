
# `copy_n`

Header File: `Kokkos_StdAlgorithms.hpp`

```c++
namespace Kokkos{
namespace Experimental{

template <class ExecutionSpace, class InputIteratorType, class SizeType, class OutputIteratorType>
OutputIteratorType copy_n(const ExecutionSpace& exespace,                    (1)
                          InputIteratorType first_from,
                          SizeType n,
                          OutputIteratorType first_to);

template <class ExecutionSpace, class InputIteratorType, class SizeType, class OutputIteratorType>
OutputIteratorType copy_n(const std::string & label,
                          const ExecutionSpace& exespace,                    (2)
                          InputIteratorType first_from,
                          SizeType n,
                          OutputIteratorType first_to);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class SizeType,
  class DataType2, class... Properties2
>
auto copy_n(const ExecutionSpace& exespace,                                  (3)
            const Kokkos::View<DataType1, Properties1...>& view_from,
            SizeType n,
            const Kokkos::View<DataType2, Properties2...>& view_to);

template <
  class ExecutionSpace,
  class DataType1, class... Properties1,
  class SizeType,
  class DataType2, class... Properties2
>
auto copy_n(const std::string& label, const ExecutionSpace& exespace,        (4)
            const Kokkos::View<DataType1, Properties1...>& view_from,
            SizeType n,
            const Kokkos::View<DataType2, Properties2...>& view_to);

} //end namespace Experimental
} //end namespace Kokkos
```

Copies the first `n` elements starting at `first_from` to
another range starting at `first_to` (overloads 1,2) or the first `n` elements
from `view_from` to `view_to` (overloads 3,4).


## Parameters and Requirements

- `exespace`, `first_from`, `first_to`, `view_from`, `view_to`:
  - same as in [`copy`](./StdCopy)
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::copy_n_if_iterator_api_default"
  - for 3, the default string is: "Kokkos::copy_n_if_view_api_default"
- `n`:
  - number of elements to copy (must be non-negative)


## Return

If `n>0`, returns an iterator to the destination element *after* the last element copied.

Otherwise, returns `first_to` (for 1,2) or `Kokkos::begin(view_to)` (for 3,4).
