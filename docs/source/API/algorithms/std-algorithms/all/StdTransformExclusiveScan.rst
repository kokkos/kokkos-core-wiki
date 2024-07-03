``transform_exclusive_scan``
============================

Header: ``Kokkos_StdAlgorithms.hpp``

Description
-----------

Transforms each element in a range or a ``view`` with ``unary_op`` then computes an exclusive
prefix scan operation using ``binary_op`` over the resulting range, with ``init`` as the initial value,
and writes the results to the range beginning at ``first_dest`` or to ``view_dest``.
"exclusive" means that the i-th input element is not included in the i-th sum.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.

Overload set accepting execution space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <
      class ExecutionSpace, class InputIteratorType,
      class OutputIteratorType, class ValueType,
      class BinaryOpType, class UnaryOpType>
   OutputIteratorType transform_exclusive_scan(const ExecutionSpace& exespace,   (1)
                                               InputIteratorType first_from,
                                               InputIteratorType last_from,
                                               OutputIteratorType first_dest,
                                               ValueType init_value,
                                               BinaryOpType binary_op,
                                               UnaryOpType unary_op);

   template <
      class ExecutionSpace, class InputIteratorType,
      class OutputIteratorType, class ValueType,
      class BinaryOpType, class UnaryOpType>
   OutputIteratorType transform_exclusive_scan(const std::string& label,         (2)
                                               const ExecutionSpace& exespace,
                                               InputIteratorType first_from,
                                               InputIteratorType last_from,
                                               OutputIteratorType first_dest,
                                               ValueType init_value,
                                               BinaryOpType binary_op,
                                               UnaryOpType unary_op);

   template <
      class ExecutionSpace,
      class DataType1, class... Properties1,
      class DataType2, class... Properties2,
      class ValueType, class BinaryOpType, class UnaryOpType>
   auto transform_exclusive_scan(const ExecutionSpace& exespace,                 (3)
                                 const Kokkos::View<DataType1, Properties1...>& view_from,
                                 const Kokkos::View<DataType2, Properties2...>& view_dest,
                                 ValueType init_value,
                                 BinaryOpType binary_op,
                                 UnaryOpType unary_op);

   template <
      class ExecutionSpace,
      class DataType1, class... Properties1,
      class DataType2, class... Properties2,
      class ValueType, class BinaryOpType, class UnaryOpType>
   auto transform_exclusive_scan(const std::string& label,                       (4)
                                 const ExecutionSpace& exespace,
                                 const Kokkos::View<DataType1, Properties1...>& view_from,
                                 const Kokkos::View<DataType2, Properties2...>& view_dest,
                                 ValueType init_value,
                                 BinaryOpType binary_op,
                                 UnaryOpType unary_op);

Overload set accepting a team handle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 4.2

.. code-block:: cpp

   template <
      class TeamHandleType, class InputIteratorType,
      class OutputIteratorType, class ValueType,
      class BinaryOpType, class UnaryOpType>
   KOKKOS_FUNCTION
   OutputIteratorType transform_exclusive_scan(const TeamHandleType& teamHandle, (5)
                                               InputIteratorType first_from,
                                               InputIteratorType last_from,
                                               OutputIteratorType first_dest,
                                               ValueType init_value,
                                               BinaryOpType binary_op,
                                               UnaryOpType unary_op);

   template <
      class TeamHandleType,
      class DataType1, class... Properties1,
      class DataType2, class... Properties2,
      class ValueType, class BinaryOpType, class UnaryOpType>
   KOKKOS_FUNCTION
   auto transform_exclusive_scan(const TeamHandleType& teamHandle,               (6)
                                 const Kokkos::View<DataType1, Properties1...>& view_from,
                                 const Kokkos::View<DataType2, Properties2...>& view_dest,
                                 ValueType init_value,
                                 BinaryOpType binary_op,
                                 UnaryOpType unary_op);

Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. |ExclusiveScan| replace:: ``exclusive_scan``
.. _ExclusiveScan: ./StdExclusiveScan.html

- ``exespace``, ``first_from``, ``first_last``, ``first_dest``, ``view_from``, ``view_dest``: same as |ExclusiveScan|_

- ``teamHandle``: team handle instance given inside a parallel region when using a TeamPolicy

- ``label``: string forwarded to internal parallel kernels for debugging purposes

  - 1: The default string is "Kokkos::transform_exclusive_scan_custom_functors_iterator_api"

  - 3: The default string is "Kokkos::transform_exclusive_scan_custom_functors_view_api"

  - NOTE: overloads accepting a team handle do not use a label internally

- ``unary_op``:

  - *unary* functor performing the desired transformation operation to an element.
    Must be valid to be called from the execution space passed, and callable with ``v`` of type
    (possible const) ``value_type``, where ``value_type`` is the value type of ``first_from``
    or value type of ``view_from``, and must not modify ``v``.

  - Must conform to:

  .. code-block:: cpp

     struct UnaryOp {
       KOKKOS_FUNCTION
       constexpr value_type operator()(const value_type & v) const {
         return /* ... */
       }
     };

Return Value
~~~~~~~~~~~~

Iterator to the element *after* the last element written.
