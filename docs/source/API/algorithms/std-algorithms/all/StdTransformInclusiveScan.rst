``transform_inclusive_scan``
============================

Header: ``Kokkos_StdAlgorithms.hpp``

Description
-----------

Transforms each element in a range or ``view_from`` with ``unary_op``, then computes an inclusive prefix scan operation using ``binary_op`` over the resulting range, with ``init_value`` as the initial value, and writes the results to the range beginning at ``first_dest`` or to ``view_dest``.

Inclusive means that the i-th input element is included in the i-th sum.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.

.. code-block:: cpp

   //
   // overload set A: no init value
   //
   template <
      class ExecutionSpace, class InputIteratorType,
      class OutputIteratorType, class BinaryOpType, class UnaryOpType>
   OutputIteratorType transform_inclusive_scan(const ExecutionSpace& exespace,      (1)
                                               InputIteratorType first_from,
                                               InputIteratorType last_from,
                                               OutputIteratorType first_dest,
                                               BinaryOpType binary_op,
                                               UnaryOpType unary_op);

   template <
      class ExecutionSpace, class InputIteratorType,
      class OutputIteratorType, class BinaryOpType, class UnaryOpType>
   OutputIteratorType transform_inclusive_scan(const std::string& label,            (2)
                                               const ExecutionSpace& exespace,
                                               InputIteratorType first_from,
                                               InputIteratorType last_from,
                                               OutputIteratorType first_dest,
                                               BinaryOpType binary_op,
                                               UnaryOpType unary_op);

   template <
      class ExecutionSpace,
      class DataType1, class... Properties1,
      class DataType2, class... Properties2,
      class BinaryOpType, class UnaryOpType>
   auto transform_inclusive_scan(const ExecutionSpace& exespace,                    (3)
                                 const Kokkos::View<DataType1, Properties1...>& view_from,
                                 const Kokkos::View<DataType2, Properties2...>& view_dest,
                                 BinaryOpType binary_op,
                                 UnaryOpType unary_op);

   template <
      class ExecutionSpace,
      class DataType1, class... Properties1,
      class DataType2, class... Properties2,
      class BinaryOpType, class UnaryOpType>
   auto transform_inclusive_scan(const std::string& label,                          (4)
                                 const ExecutionSpace& exespace,
                                 const Kokkos::View<DataType1, Properties1...>& view_from,
                                 const Kokkos::View<DataType2, Properties2...>& view_dest,
                                 BinaryOpType binary_op,
                                 UnaryOpType unary_op);

   template <
      class TeamHandleType, class InputIteratorType,
      class OutputIteratorType, class BinaryOpType, class UnaryOpType>
   KOKKOS_FUNCTION
   OutputIteratorType transform_inclusive_scan(const TeamHandleType& teamHandle,    (5)
                                               InputIteratorType first_from,
                                               InputIteratorType last_from,
                                               OutputIteratorType first_dest,
                                               BinaryOpType binary_op,
                                               UnaryOpType unary_op);

   template <
      class TeamHandleType,
      class DataType1, class... Properties1,
      class DataType2, class... Properties2,
      class BinaryOpType, class UnaryOpType>
   KOKKOS_FUNCTION
   auto transform_inclusive_scan(const TeamHandleType& teamHandle,                  (6)
                                 const Kokkos::View<DataType1, Properties1...>& view_from,
                                 const Kokkos::View<DataType2, Properties2...>& view_dest,
                                 BinaryOpType binary_op,
                                 UnaryOpType unary_op);

   //
   // overload set B: init value
   //
   template <
      class ExecutionSpace, class InputIteratorType,
      class OutputIteratorType, class BinaryOpType, class UnaryOpType,
      class ValueType>
   OutputIteratorType transform_inclusive_scan(const ExecutionSpace& exespace,      (7)
                                               InputIteratorType first_from,
                                               InputIteratorType last_from,
                                               OutputIteratorType first_dest,
                                               BinaryOpType binary_op,
                                               UnaryOpType unary_op,
                                               ValueType init_value);

   template <
      class ExecutionSpace, class InputIteratorType,
      class OutputIteratorType, class BinaryOpType, class UnaryOpType,
      class ValueType>
   OutputIteratorType transform_inclusive_scan(const std::string& label,            (8)
                                               const ExecutionSpace& exespace,
                                               InputIteratorType first_from,
                                               InputIteratorType last_from,
                                               OutputIteratorType first_dest,
                                               BinaryOpType binary_op,
                                               UnaryOpType unary_op,
                                               ValueType init_value);

   template <
      class ExecutionSpace,
      class DataType1, class... Properties1,
      class DataType2, class... Properties2,
      class BinaryOpType, class UnaryOpType, class ValueType>
   auto transform_inclusive_scan(const ExecutionSpace& exespace,                    (9)
                                 const Kokkos::View<DataType1, Properties1...>& view_from,
                                 const Kokkos::View<DataType2, Properties2...>& view_dest,
                                 BinaryOpType binary_op,
                                 UnaryOpType unary_op,
                                 ValueType init_value);

   template <
      class ExecutionSpace,
      class DataType1, class... Properties1,
      class DataType2, class... Properties2,
      class BinaryOpType, class UnaryOpType, class ValueType>
   auto transform_inclusive_scan(const std::string& label,                          (10)
                                 const ExecutionSpace& exespace,
                                 const Kokkos::View<DataType1, Properties1...>& view_from,
                                 const Kokkos::View<DataType2, Properties2...>& view_dest,
                                 BinaryOpType binary_op,
                                 UnaryOpType unary_op,
                                 ValueType init_value);

   template <
      class TeamHandleType, class InputIteratorType,
      class OutputIteratorType, class BinaryOpType, class UnaryOpType,
      class ValueType>
   KOKKOS_FUNCTION
   OutputIteratorType transform_inclusive_scan(const TeamHandleType& teamHandle,    (11)
                                               InputIteratorType first_from,
                                               InputIteratorType last_from,
                                               OutputIteratorType first_dest,
                                               BinaryOpType binary_op,
                                               UnaryOpType unary_op,
                                               ValueType init_value);

   template <
      class TeamHandleType,
      class DataType1, class... Properties1,
      class DataType2, class... Properties2,
      class BinaryOpType, class UnaryOpType, class ValueType>
   KOKKOS_FUNCTION
   auto transform_inclusive_scan(const TeamHandleType& teamHandle,                  (12)
                                 const Kokkos::View<DataType1, Properties1...>& view_from,
                                 const Kokkos::View<DataType2, Properties2...>& view_dest,
                                 BinaryOpType binary_op,
                                 UnaryOpType unary_op,
                                 ValueType init_value);

Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. |TransformExclusiveScan| replace:: ``transform_exclusive_scan``
.. _TransformExclusiveScan: ./StdTransformExclusiveScan.html

- ``exespace``, ``first_from``, ``first_last``, ``first_dest``, ``view_from``, ``view_dest``, ``init_value``, ``bin_op``, ``unary_op``: same as |TransformExclusiveScan|_

- ``teamHandle``: team handle instance given inside a parallel region when using a TeamPolicy

- ``label``: string forwarded to internal parallel kernels for debugging purposes

  - 1,7: The default string is "Kokkos::transform_inclusive_scan_custom_functors_iterator_api"

  - 3,9: The default string is "Kokkos::transform_inclusive_scan_custom_functors_view_api"

Return Value
~~~~~~~~~~~~

Iterator to the element *after* the last element written.