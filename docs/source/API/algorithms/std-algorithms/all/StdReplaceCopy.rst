
``replace_copy``
=================

Header: ``<Kokkos_StdAlgorithms.hpp>``

Description
-----------

Copies the elements from a given range ``[first_from, last_from)`` to another range
beginning at ``first_to``, while replacing all elements that equal ``old_value`` 
with ``new_value``.
The overload taking a ``View`` uses the ``begin`` and ``end`` iterators of the ``View``.
Comparison between elements is done using ``operator==``.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.


Overload set accepting execution space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <class ExecutionSpace, class InputIteratorType, class OutputIteratorType, class T>
   OutputIteratorType replace_copy(const ExecutionSpace& exespace,                (1)
                                   InputIteratorType first_from,
                                   InputIteratorType last_from,
                                   OutputIteratorType first_to,
                                   const T& old_value, const T& new_value);

   template <class ExecutionSpace, class InputIteratorType, class OutputIteratorType, class T>
   OutputIteratorType replace_copy(const std::string& label,                      (2)
                                   const ExecutionSpace& exespace,
                                   OutputIteratorType first_to,
                                   const T& old_value, const T& new_value);

   template <
     class ExecutionSpace,
     class DataType1, class... Properties1,
     class DataType2, class... Properties2,
     class T
   >
   auto replace_copy(const ExecutionSpace& exespace,                              (3)
                     const Kokkos::View<DataType1, Properties1...>& view_from,
                     const Kokkos::View<DataType2, Properties2...>& view_to,
                     const T& old_value, const T& new_value);

   template <
     class ExecutionSpace,
     class DataType1, class... Properties1,
     class DataType2, class... Properties2,
     class T
   >
   auto replace_copy(const std::string& label,
                     const ExecutionSpace& exespace,                              (4)
                     const Kokkos::View<DataType1, Properties1...>& view_from,
                     const Kokkos::View<DataType2, Properties2...>& view_to,
                     const T& old_value, const T& new_value);

Overload set accepting a team handle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 4.2

.. code-block:: cpp

   template <class TeamHandleType, class InputIterator, class OutputIterator,
             class ValueType>
   KOKKOS_FUNCTION
   OutputIterator replace_copy(const TeamHandleType& teamHandle,                   (5)
                               InputIterator first_from, InputIterator last_from,
                               OutputIterator first_dest,
                               const ValueType& old_value, const ValueType& new_value);

   template <
       class TeamHandleType, class DataType1, class... Properties1,
       class DataType2, class... Properties2, class ValueType, int>
   KOKKOS_FUNCTION
   auto replace_copy(const TeamHandleType& teamHandle,                             (6)
                     const Kokkos::View<DataType1, Properties1...>& view_from,
                     const Kokkos::View<DataType2, Properties2...>& view_dest,
                     const ValueType& old_value, const ValueType& new_value);


Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``exespace``: execution space instance

- ``teamHandle``: team handle instance given inside a parallel region when using a TeamPolicy

- ``label``: used to name the implementation kernels for debugging purposes

  - for 1, the default string is: "Kokkos::replace_copy_iterator_api_default"

  - for 3, the default string is: "Kokkos::replace_copy_view_api_default"

  - NOTE: overloads accepting a team handle do not use a label internally

- ``first_from, last_from``: range of elements to copy from

  - must be *random access iterators*

  - must represent a valid range, i.e., ``last_from >= first_from`` (checked in debug mode)

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``first_to``: beginning of the range to copy to

  - must be a *random access iterator*

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``view_from``, ``view_to``:

  - source and destination views

  - must be rank-1, and have ``LayoutLeft``, ``LayoutRight``, or ``LayoutStride``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``old_value``, ``new_value``: self-explanatory


Return Value
~~~~~~~~~~~~

Iterator to the element *after* the last element copied.
