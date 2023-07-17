``partition_point``
===================

Header: ``Kokkos_StdAlgorithms.hpp``

Description
-----------

Examines a range or ``view`` and locates the first element that does not satisfy ``pred``. Assumes the range (or the view) already to be partitioned.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.

.. code-block:: cpp

   //
   // overload set accepting execution space
   //
   template <class ExecutionSpace, class IteratorType, class UnaryPredicate>
   IteratorType partition_point(const ExecutionSpace& exespace,                   (1)
                                IteratorType first, IteratorType last,
                                UnaryPredicate pred);

   template <class ExecutionSpace, class IteratorType, class UnaryPredicate>
   IteratorType partition_point(const std::string& label,                         (2)
                                const ExecutionSpace& exespace,
                                IteratorType first, IteratorType last,
                                UnaryPredicate pred);

   template <
      class ExecutionSpace, class UnaryPredicate,
      class DataType, class... Properties>
   auto partition_point(const std::string& label,                                 (3)
                        const ExecutionSpace& exespace,
                        const Kokkos::View<DataType, Properties...>& view,
                        UnaryPredicate pred);

   template <
      class ExecutionSpace, class UnaryPredicate,
      class DataType, class... Properties>
   auto partition_point(const ExecutionSpace& exespace,                           (4)
                        const Kokkos::View<DataType, Properties...>& view,
                        UnaryPredicate pred);

   //
   // overload set accepting a team handle
   //
   template <class TeamHandleType, class IteratorType, class UnaryPredicate>
   KOKKOS_FUNCTION
   IteratorType partition_point(const TeamHandleType& teamHandle,                 (5)
                                IteratorType first, IteratorType last,
                                UnaryPredicate pred);

   template <
      class TeamHandleType, class UnaryPredicate,
      class DataType, class... Properties>
   KOKKOS_FUNCTION
   auto partition_point(const TeamHandleType& teamHandle,                         (6)
                        const Kokkos::View<DataType, Properties...>& view,
                        UnaryPredicate pred);

Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. |IsPartitioned| replace:: ``is_partitioned``
.. _IsPartitioned: ./StdIsPartitioned.html

- ``exespace``, ``first``, ``last``, ``view``, ``pred``: same as in |IsPartitioned|_

- ``teamHandle``: team handle instance given inside a parallel region when using a TeamPolicy

- ``label``: string forwarded to internal parallel kernels for debugging purposes

  - 1: The default string is "Kokkos::partitioned_point_iterator_api_default"

  - 4: The default string is "Kokkos::partition_point_view_api_default"

  - NOTE: overloads accepting a team handle do not use a label internally

Return Value
~~~~~~~~~~~~

Iterator to the element *after* the last element in the first partition, or ``last`` if all elements satisfy ``pred``.