``exclusive_scan``
==================

Header: ``Kokkos_StdAlgorithms.hpp``

Description
-----------

Computes an exclusive prefix:

- *sum* for the range ``[first_from, last_from)``

- scan using the binary functor ``bin_op`` to combine two elements for the range ``[first_from, last_from)``

or ``view_from``, using ``init`` as the initial value, and writes the results to the range beginning at ``first_dest`` or to ``view_dest``.

Exclusive means that the i-th input element is not included in the i-th sum.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.

.. code-block:: cpp

   //
   // overload set A
   //
   template <
      class ExecutionSpace, class InputIteratorType,
      class OutputIteratorType, class ValueType>
   OutputIteratorType exclusive_scan(const ExecutionSpace& exespace,             (1)
                                     InputIteratorType first,
                                     InputIteratorType last,
                                     OutputIteratorType first_dest,
                                     ValueType init_value);

   template <
      class ExecutionSpace, class InputIteratorType,
      class OutputIteratorType, class ValueType>
   OutputIteratorType exclusive_scan(const std::string& label,                   (2)
                                     const ExecutionSpace& exespace,
                                     InputIteratorType first,
                                     InputIteratorType last,
                                     OutputIteratorType first_dest,
                                     ValueType init_value);

   template <
      class ExecutionSpace,
      class DataType1, class... Properties1,
      class DataType2, class... Properties2,
      class ValueType>
   auto exclusive_scan(const ExecutionSpace& exespace,                           (3)
                       const Kokkos::View<DataType1, Properties1...>& view_from,
                       const Kokkos::View<DataType2, Properties2...>& view_dest,
                       ValueType init_value);

   template <
      class ExecutionSpace,
      class DataType1, class... Properties1,
      class DataType2, class... Properties2,
      class ValueType>
   auto exclusive_scan(const std::string& label, const ExecutionSpace& exespace, (4)
                       const Kokkos::View<DataType1, Properties1...>& view_from,
                       const Kokkos::View<DataType2, Properties2...>& view_dest,
                       ValueType init_value);

   template <
      class TeamHandleType, class InputIteratorType,
      class OutputIteratorType, class ValueType>
   KOKKOS_FUNCTION
   OutputIteratorType exclusive_scan(const TeamHandleType& teamHandle,           (5)
                                     InputIteratorType first,
                                     InputIteratorType last,
                                     OutputIteratorType first_dest,
                                     ValueType init_value);

   template <
      class TeamHandleType,
      class DataType1, class... Properties1,
      class DataType2, class... Properties2,
      class ValueType>
   KOKKOS_FUNCTION
   auto exclusive_scan(const TeamHandleType& teamHandle,                         (6)
                       const Kokkos::View<DataType1, Properties1...>& view_from,
                       const Kokkos::View<DataType2, Properties2...>& view_dest,
                       ValueType init_value);

   //
   // overload set B
   //
   template <
      class ExecutionSpace, class InputIteratorType,
      class OutputIteratorType, class ValueType, class BinaryOpType>
   OutputIteratorType exclusive_scan(const ExecutionSpace& exespace,             (7)
                                     InputIteratorType first,
                                     InputIteratorType last,
                                     OutputIteratorType first_dest,
                                     ValueType init_value, BinaryOpType bop);

   template <
      class ExecutionSpace, class InputIteratorType,
      class OutputIteratorType, class ValueType, class BinaryOpType>
   OutputIteratorType exclusive_scan(const std::string& label,                   (8)
                                     const ExecutionSpace& exespace,
                                     InputIteratorType first,
                                     InputIteratorType last,
                                     OutputIteratorType first_dest,
                                     ValueType init_value,
                                     BinaryOpType bop);

   template <
      class ExecutionSpace,
      class DataType1, class... Properties1,
      class DataType2, class... Properties2,
      class ValueType, class BinaryOpType>
   auto exclusive_scan(const ExecutionSpace& exespace,                           (9)
                       const Kokkos::View<DataType1, Properties1...>& view_from,
                       const Kokkos::View<DataType2, Properties2...>& view_dest,
                       ValueType init_value, BinaryOpType bop);

   template <
      class ExecutionSpace,
      class DataType1, class... Properties1,
      class DataType2, class... Properties2,
      class ValueType, class BinaryOpType>
   auto exclusive_scan(const std::string& label,                                 (10)
                       const ExecutionSpace& exespace,
                       const Kokkos::View<DataType1, Properties1...>& view_from,
                       const Kokkos::View<DataType2, Properties2...>& view_dest,
                       ValueType init_value, BinaryOpType bop);

   template <
      class TeamHandleType, class InputIteratorType,
      class OutputIteratorType, class ValueType, class BinaryOpType>
   KOKKOS_FUNCTION
   OutputIteratorType exclusive_scan(const TeamHandleType& teamHandle,           (11)
                                     InputIteratorType first,
                                     InputIteratorType last,
                                     OutputIteratorType first_dest,
                                     ValueType init_value,
                                     BinaryOpType bop);

   template <
      class TeamHandleType,
      class DataType1, class... Properties1,
      class DataType2, class... Properties2,
      class ValueType, class BinaryOpType>
   KOKKOS_FUNCTION
   auto exclusive_scan(const TeamHandleType& teamHandle,                         (12)
                       const Kokkos::View<DataType1, Properties1...>& view_from,
                       const Kokkos::View<DataType2, Properties2...>& view_dest,
                       ValueType init_value, BinaryOpType bop);

Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``exespace``: execution space instance

- ``teamHandle``: team handle instance given inside a parallel region when using a TeamPolicy

- ``label``: string forwarded to internal parallel kernels for debugging purposes

  - 1: The default string is "Kokkos::exclusive_scan_default_functors_iterator_api"

  - 3: The default string is "Kokkos::exclusive_scan_default_functors_view_api"

  - 7: The default string is "Kokkos::exclusive_scan_custom_functors_iterator_api"

  - 9: The default string is "Kokkos::exclusive_scan_custom_functors_view_api"

  - NOTE: overloads accepting a team handle do not use a label internally

- ``first_from``, ``last_from``, ``first_dest``: range of elements to read from (``*_from``) and write to (``first_dest``)

  - must be *random access iterators*, e.g., returned from ``Kokkos::Experimental::(c)begin/(c)end``

  - must represent a valid range, i.e., ``last_from >= first_from``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``view_from``, ``view_dest``: views to read elements from ``view_from`` and write to ``view_dest``

  - must be rank-1, and have ``LayoutLeft``, ``LayoutRight``, or ``LayoutStride``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``bin_op``:

  - *binary* functor representing the operation to combine pair of elements. Must be valid to be called from the execution space passed, and callable with two arguments ``a,b`` of type (possible const) ``value_type``, where ``value_type`` is the value type of ``InputIteratorType`` or the value type of ``view_from``, and must not modify ``a,b``.

  - must conform to:

  .. code-block:: cpp

     struct BinaryOp
     {
        KOKKOS_INLINE_FUNCTION
        return_type operator()(const value_type & a,
                               const value_type & b) const {
           return /* ... */;
        }
     };

  The return type ``return_type`` must be such that an object of type ``OutputIteratorType``
  or an object of type ``value_type`` where ``value_type`` is the
  value type of ``view_dest`` can be dereferenced and assigned a value of type ``return_type``.

Return Value
~~~~~~~~~~~~

Iterator to the element *after* the last element written.