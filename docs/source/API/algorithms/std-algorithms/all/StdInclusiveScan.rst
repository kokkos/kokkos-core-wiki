``inclusive_scan``
==================

Header: ``Kokkos_StdAlgorithms.hpp``

Description
-----------

Computes an inclusive prefix scan over a range or a ``view_from`` using the binary op ``bin_op`` to combine two elements,
and ``init`` as the initial value, and writes the results to the range beginning at ``first_dest`` or to ``view_dest``.
Inclusive means that the i-th input element is included in the i-th sum.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.

Overload set accepting execution space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <class ExecutionSpace, class InputIteratorType, class OutputIteratorType>
   OutputIteratorType inclusive_scan(const ExecutionSpace& exespace,                   (1)
                                     InputIteratorType first_from,
                                     InputIteratorType last_from,
                                     OutputIteratorType first_dest);

   template <class ExecutionSpace, class InputIteratorType, class OutputIteratorType>
   OutputIteratorType inclusive_scan(const std::string& label,                         (2)
                                     const ExecutionSpace& exespace,
                                     InputIteratorType first_from,
                                     InputIteratorType last_from,
                                     OutputIteratorType first_dest);

   template <
      class ExecutionSpace,
      class DataType1, class... Properties1,
      class DataType2, class... Properties2>
   auto inclusive_scan(const ExecutionSpace& exespace,                                 (3)
                       const Kokkos::View<DataType1, Properties1...>& view_from,
                       const Kokkos::View<DataType2, Properties2...>& view_dest);

   template <
      class ExecutionSpace,
      class DataType1, class... Properties1,
      class DataType2, class... Properties2>
   auto inclusive_scan(const std::string& label,                                       (4)
                       const ExecutionSpace& exespace,
                       const Kokkos::View<DataType1, Properties1...>& view_from,
                       const Kokkos::View<DataType2, Properties2...>& view_dest);

   template <class TeamHandleType, class InputIteratorType, class OutputIteratorType>
   KOKKOS_FUNCTION
   OutputIteratorType inclusive_scan(const TeamHandleType& teamHandle,                 (5)
                                     InputIteratorType first_from,
                                     InputIteratorType last_from,
                                     OutputIteratorType first_dest);

   template <
      class TeamHandleType,
      class DataType1, class... Properties1,
      class DataType2, class... Properties2>
   KOKKOS_FUNCTION
   auto inclusive_scan(const TeamHandleType& teamHandle,                               (6)
                       const Kokkos::View<DataType1, Properties1...>& view_from,
                       const Kokkos::View<DataType2, Properties2...>& view_dest);

   template <
      class ExecutionSpace, class InputIteratorType,
      class OutputIteratorType, class BinaryOp>
   OutputIteratorType inclusive_scan(const ExecutionSpace& exespace,                   (7)
                                     InputIteratorType first_from,
                                     InputIteratorType last_from,
                                     OutputIteratorType first_dest,
                                     BinaryOp binary_op);

   template <
      class ExecutionSpace, class InputIteratorType,
      class OutputIteratorType, class BinaryOp>
   OutputIteratorType inclusive_scan(const std::string& label,                         (8)
                                     const ExecutionSpace& exespace,
                                     InputIteratorType first_from,
                                     InputIteratorType last_from,
                                     OutputIteratorType first_dest,
                                     BinaryOp binary_op);

   template <
      class ExecutionSpace,
      class DataType1, class... Properties1,
      class DataType2, class... Properties2,
      class BinaryOp>
   auto inclusive_scan(const ExecutionSpace& exespace,                                 (9)
                       const Kokkos::View<DataType1, Properties1...>& view_from,
                       const Kokkos::View<DataType2, Properties2...>& view_dest,
                       BinaryOp binary_op);

   template <
      class ExecutionSpace,
      class DataType1, class... Properties1,
      class DataType2, class... Properties2,
      class BinaryOp>
   auto inclusive_scan(const std::string& label,                                       (10)
                       const ExecutionSpace& exespace,
                       const Kokkos::View<DataType1, Properties1...>& view_from,
                       const Kokkos::View<DataType2, Properties2...>& view_dest,
                       BinaryOp binary_op);

   template <
      class ExecutionSpace,
      class InputIteratorType, class OutputIteratorType,
      class BinaryOp, class ValueType>
   OutputIteratorType inclusive_scan(const ExecutionSpace& exespace,                   (11)
                                     InputIteratorType first_from,
                                     InputIteratorType last_from,
                                     OutputIteratorType first_dest,
                                     BinaryOp binary_op,
                                     ValueType init_value);

   template <
      class ExecutionSpace, class InputIteratorType,
      class OutputIteratorType, class BinaryOp, class ValueType>
   OutputIteratorType inclusive_scan(const std::string& label,                         (12)
                                     const ExecutionSpace& exespace,
                                     InputIteratorType first_from,
                                     InputIteratorType last_from,
                                     OutputIteratorType first_dest,
                                     BinaryOp binary_op,
                                     ValueType init_value);

   template <
      class ExecutionSpace,
      class DataType1, class... Properties1,
      class DataType2, class... Properties2,
      class BinaryOp, class ValueType>
   auto inclusive_scan(const ExecutionSpace& exespace,                                 (13)
                       const Kokkos::View<DataType1, Properties1...>& view_from,
                       const Kokkos::View<DataType2, Properties2...>& view_dest,
                       BinaryOp binary_op,
                       ValueType init_value);

   template <
      class ExecutionSpace,
      class DataType1, class... Properties1,
      class DataType2, class... Properties2,
      class BinaryOp, class ValueType>
   auto inclusive_scan(const std::string& label,                                       (14)
                       const ExecutionSpace& exespace,
                       const Kokkos::View<DataType1, Properties1...>& view_from,
                       const Kokkos::View<DataType2, Properties2...>& view_dest,
                       BinaryOp binary_op,
                       ValueType init_value);


Overload set accepting a team handle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 4.2

.. code-block:: cpp

   template <
      class TeamHandleType, class InputIteratorType,
      class OutputIteratorType, class BinaryOp>
   KOKKOS_FUNCTION
   OutputIteratorType inclusive_scan(const TeamHandleType& teamHandle,                 (15)
                                     InputIteratorType first_from,
                                     InputIteratorType last_from,
                                     OutputIteratorType first_dest,
                                     BinaryOp binary_op);

   template <
      class TeamHandleType,
      class DataType1, class... Properties1,
      class DataType2, class... Properties2,
      class BinaryOp>
   KOKKOS_FUNCTION
   auto inclusive_scan(const TeamHandleType& teamHandle,                               (16)
                       const Kokkos::View<DataType1, Properties1...>& view_from,
                       const Kokkos::View<DataType2, Properties2...>& view_dest,
                       BinaryOp binary_op);


   template <
      class TeamHandleType, class InputIteratorType,
      class OutputIteratorType, class BinaryOp, class ValueType>
   KOKKOS_FUNCTION
   OutputIteratorType inclusive_scan(const TeamHandleType& teamHandle,                 (17)
                                     InputIteratorType first_from,
                                     InputIteratorType last_from,
                                     OutputIteratorType first_dest,
                                     BinaryOp binary_op,
                                     ValueType init_value);

   template <
      class TeamHandleType,
      class DataType1, class... Properties1,
      class DataType2, class... Properties2,
      class BinaryOp, class ValueType>
   KOKKOS_FUNCTION
   auto inclusive_scan(const TeamHandleType& teamHandle,                               (18)
                       const Kokkos::View<DataType1, Properties1...>& view_from,
                       const Kokkos::View<DataType2, Properties2...>& view_dest,
                       BinaryOp binary_op,
                       ValueType init_value);

Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. |ExclusiveScan| replace:: ``exclusive_scan``
.. _ExclusiveScan: ./StdExclusiveScan.html

- ``exespace``, ``first_from``, ``first_last``, ``first_dest``, ``view_from``, ``view_dest``, ``bin_op``: same as in |ExclusiveScan|_

- ``teamHandle``: team handle instance given inside a parallel region when using a TeamPolicy

- ``label``: string forwarded to internal parallel kernels for debugging purposes

  - 1: The default string is "Kokkos::inclusive_scan_default_functors_iterator_api"

  - 3: The default string is "Kokkos::inclusive_scan_default_functors_view_api"

  - 7, 13: The default string is "Kokkos::inclusive_scan_custom_functors_iterator_api"

  - 9, 15: The default string is "Kokkos::inclusive_scan_custom_functors_view_api"

  - NOTE: overloads accepting a team handle do not use a label internally

Return Value
~~~~~~~~~~~~

Iterator to the element *after* the last element written.
