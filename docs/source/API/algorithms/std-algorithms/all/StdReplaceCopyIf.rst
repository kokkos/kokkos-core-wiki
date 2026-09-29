
``replace_copy_if``
====================

Header: ``<Kokkos_StdAlgorithms.hpp>``

Description
-----------

Copies the elements from range ``[first_from, last_from)`` to another range
beginning at ``first_to`` while replacing all elements for which ``pred`` returns ``true`` with ``new_value``.
The overload taking a ``View`` uses the ``begin`` and ``end`` iterators of the ``View``.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.


Overload set accepting execution space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <
     class ExecutionSpace,
     class InputIteratorType, class OutputIteratorType,
     class UnaryPredicateType, class T
   >
   OutputIteratorType replace_copy_if(const ExecutionSpace& exespace,               (1)
                                      InputIteratorType first_from,
                                      InputIteratorType last_from,
                                      OutputIteratorType first_to,
                                      UnaryPredicateType pred, const T& new_value);

   template <
     class ExecutionSpace,
     class InputIteratorType,  class OutputIteratorType,
     class UnaryPredicateType, class T
   >
   OutputIteratorType replace_copy_if(const std::string& label,                     (2)
                                      const ExecutionSpace& exespace,
                                      InputIteratorType first_from,
                                      InputIteratorType last_from,
                                      OutputIteratorType first_to,
                                      UnaryPredicateType pred, const T& new_value);

   template <
     class ExecutionSpace,
     class DataType1, class... Properties1,
     class DataType2, class... Properties2,
     class UnaryPredicateType, class T
   >
   auto replace_copy_if(const ExecutionSpace& exespace,                             (3)
                        const Kokkos::View<DataType1, Properties1...>& view_from,
                        const Kokkos::View<DataType2, Properties2...>& view_to,
                        UnaryPredicateType pred, const T& new_value);

   template <
     class ExecutionSpace,
     class DataType1, class... Properties1,
     class DataType2, class... Properties2,
     class UnaryPredicateType, class T
   >
     auto replace_copy_if(const std::string& label,                                 (4)
                          const ExecutionSpace& exespace,
                          const Kokkos::View<DataType1, Properties1...>& view_from,
                          const Kokkos::View<DataType2, Properties2...>& view_to,
                          UnaryPredicateType pred, const T& new_value);

Overload set accepting a team handle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 4.2

.. code-block:: cpp

   template <class TeamHandleType, class InputIterator, class OutputIterator,
             class PredicateType, class ValueType>
   KOKKOS_FUNCTION
   OutputIterator
   replace_copy_if(const TeamHandleType& teamHandle, InputIterator first_from,      (5)
                   InputIterator last_from, OutputIterator first_dest,
                   PredicateType pred, const ValueType& new_value);

   template <class TeamHandleType, class DataType1, class... Properties1,
             class DataType2, class... Properties2, class PredicateType,
             class ValueType, int>
   KOKKOS_FUNCTION
   auto replace_copy_if(const TeamHandleType& teamHandle,                           (6)
                        const ::Kokkos::View<DataType1, Properties1...>& view_from, 
                        const ::Kokkos::View<DataType2, Properties2...>& view_dest,
                        PredicateType pred, const ValueType& new_value);


Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``exespace``, ``teamHandle``, ``first_from``, ``last_from``, ``first_to``, ``view_from``, ``view_to``, ``new_value``:

  - same as in [``replace_copy``](./StdReplaceCopy)

- ``label``: used to name the implementation kernels for debugging purposes

  - for 1, the default string is: "Kokkos::replace_copy_if_iterator_api_default"

  - for 3, the default string is: "Kokkos::replace_copy_if_view_api_default"

  - NOTE: overloads accepting a team handle do not use a label internally

- ``pred``: unary predicate returning ``true`` for the required element. 

  ``pred(v)`` must be valid to be called from the execution space passed, or
  the execution space associated with the team handle, and convertible to bool for every 
  argument ``v`` of type (possible const) ``value_type``, where ``value_type`` 
  is the value type of ``InputIteratorType`` or of ``view_from``, and must not modify ``v``.

  - should have the same API as that shown for [``replace_if``](./StdReplaceIf)

Return Value
~~~~~~~~~~~~

Iterator to the element *after* the last element copied.
