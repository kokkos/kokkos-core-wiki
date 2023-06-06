``copy_n``
==========

Header: `<Kokkos_StdAlgorithms.hpp>`

Description
-----------

Copies the first `n` elements starting at `first_from` to
another range starting at `first_to` (overloads 1,2,5) or the first `n` elements
from `view_from` to `view_to` (overloads 3,4,6).

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.

.. code-block:: cpp

  //
  // overload set accepting an execution space
  //
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

  //
  // overload set accepting a team handle
  //
  template <class TeamHandleType, class InputIterator, class Size,
          class OutputIterator>
  OutputIterator copy_n(const TeamHandleType& teamHandle, InputIterator first, (5)
                        Size count, OutputIterator result);

  template <
    class TeamHandleType, class DataType1, class... Properties1, class Size,
    class DataType2, class... Properties2>
  auto copy_n(                                                                 (6)
    const TeamHandleType& teamHandle,
    const ::Kokkos::View<DataType1, Properties1...>& source, Size count,
    ::Kokkos::View<DataType2, Properties2...>& dest);


Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `exespace`, `first_from`, `first_to`, `view_from`, `view_to`:
  - same as in [`copy`](./StdCopy)
- `teamHandle`:
  -  team handle instance given inside a parallel region when using a TeamPolicy
  - NOTE: overloads accepting a team handle do not use a label internally
- `label`:
  - used to name the implementation kernels for debugging purposes
  - for 1, the default string is: "Kokkos::copy_n_if_iterator_api_default"
  - for 3, the default string is: "Kokkos::copy_n_if_view_api_default"
- `n`:
  - number of elements to copy (must be non-negative)


Return Value
~~~~~~~~~~~~

If `n>0`, returns an iterator to the destination element *after* the last element copied.

Otherwise, returns `first_to` (for 1,2) or `Kokkos::begin(view_to)` (for 3,4).
