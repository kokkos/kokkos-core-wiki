``adjacent_difference``
=======================

Header: ``Kokkos_StdAlgorithms.hpp``

Description
-----------

First, a copy of ``*first_from`` is written to ``*first_dest``, or a copy of ``view_from(0)`` is written to ``view_dest(0)``.
Second, it computes the *difference* or calls the binary functor between the second and the first of each adjacent pair of elements of a range or in ``view_from``, and writes them to the range beginning at ``first_dest + 1``, or ``view_dest``.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.

.. code-block:: cpp

   //
   // overload set accepting execution space
   //
   template <
      class ExecutionSpace,
      class InputIteratorType, class OutputIteratorType>
   OutputIteratorType adjacent_difference(const ExecutionSpace& exespace,                  (1)
                                          InputIteratorType first_from,
                                          InputIteratorType last_from,
                                          OutputIteratorType first_dest);

   template <
      class ExecutionSpace,
      class InputIteratorType, class OutputIteratorType,
      class BinaryOp>
   OutputIteratorType adjacent_difference(const ExecutionSpace& exespace,                  (2)
                                          InputIteratorType first_from,
                                          InputIteratorType last_from,
                                          OutputIteratorType first_dest,
                                          BinaryOp bin_op);

   template <
      class ExecutionSpace,
      class InputIteratorType, class OutputIteratorType>
   OutputIteratorType adjacent_difference(const std::string& label,                        (3)
                                          const ExecutionSpace& exespace,
                                          InputIteratorType first_from,
                                          InputIteratorType last_from,
                                          OutputIteratorType first_dest);

   template <
      class ExecutionSpace,
      class InputIteratorType, class OutputIteratorType,
      class BinaryOp>
   OutputIteratorType adjacent_difference(const std::string& label,                        (4)
                                          const ExecutionSpace& exespace,
                                          InputIteratorType first_from,
                                          InputIteratorType last_from,
                                          OutputIteratorType first_dest,
                                          BinaryOp bin_op);

   //
   // overload set accepting views
   //
   template <
      class ExecutionSpace,
      class DataType1, class... Properties1,
      class DataType2, class... Properties2>
   auto adjacent_difference(const ExecutionSpace& exespace,                                (5)
                            const Kokkos::View<DataType1, Properties1...>& view_from,
                            const Kokkos::View<DataType2, Properties2...>& view_dest);

   template <
      class ExecutionSpace,
      class DataType1, class... Properties1,
      class DataType2, class... Properties2,
      class BinaryOp>
   auto adjacent_difference(const ExecutionSpace& exespace,                                (6)
                            const Kokkos::View<DataType1, Properties1...>& view_from,
                            const Kokkos::View<DataType2, Properties2...>& view_dest,
                            BinaryOp bin_op);

   template <
      class ExecutionSpace,
      class DataType1, class... Properties1,
      class DataType2, class... Properties2>
   auto adjacent_difference(const std::string& label,                                      (7)
                            const ExecutionSpace& exespace,
                            const Kokkos::View<DataType1, Properties1...>& view_from,
                            const Kokkos::View<DataType2, Properties2...>& view_dest);

   template <
      class ExecutionSpace,
      class DataType1, class... Properties1,
      class DataType2, class... Properties2,
      class BinaryOp>
   auto adjacent_difference(const std::string& label,                                      (8)
                            const ExecutionSpace& exespace,
                            const Kokkos::View<DataType1, Properties1...>& view_from,
                            const Kokkos::View<DataType2, Properties2...>& view_dest,
                            BinaryOp bin_op);

   //
   // overload set accepting a team handle
   //
   template <
      class TeamHandleType,
      class InputIteratorType, class OutputIteratorType>
   KOKKOS_FUNCTION
   OutputIteratorType adjacent_difference(const TeamHandleType& teamHandle,                (9)
                                          InputIteratorType first_from,
                                          InputIteratorType last_from,
                                          OutputIteratorType first_dest);

   template <
      class TeamHandleType,
      class InputIteratorType, class OutputIteratorType,
      class BinaryOp>
   KOKKOS_FUNCTION
   OutputIteratorType adjacent_difference(const TeamHandleType& teamHandle,                (10)
                                          InputIteratorType first_from,
                                          InputIteratorType last_from,
                                          OutputIteratorType first_dest,
                                          BinaryOp bin_op);

   template <
      class TeamHandleType,
      class DataType1, class... Properties1,
      class DataType2, class... Properties2>
   KOKKOS_FUNCTION
   auto adjacent_difference(const TeamHandleType& teamHandle,                              (11)
                            const Kokkos::View<DataType1, Properties1...>& view_from,
                            const Kokkos::View<DataType2, Properties2...>& view_dest);

   template <
      class TeamHandleType,
      class DataType1, class... Properties1,
      class DataType2, class... Properties2,
      class BinaryOp>
   KOKKOS_FUNCTION
   auto adjacent_difference(const TeamHandleType& teamHandle,                              (12)
                            const Kokkos::View<DataType1, Properties1...>& view_from,
                            const Kokkos::View<DataType2, Properties2...>& view_dest,
                            BinaryOp bin_op);

Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``exespace``: execution space instance

- ``teamHandle``: team handle instance given inside a parallel region when using a TeamPolicy

- ``label``: string forwarded to internal parallel kernels for debugging purposes

  - 1 & 2: The default string is "Kokkos::adjacent_difference_iterator_api"

  - 5 & 6: The default string is "Kokkos::adjacent_difference_view_api"

  - NOTE: overloads accepting a team handle do not use a label internally

- ``first_from``, ``last_from``, ``first_dest``: range of elements to read from ``*_from`` and write to ``first_dest``

  - must be *random access iterators*, e.g., returned from ``Kokkos::Experimental::(c)begin/(c)end``

  - must represent a valid range, i.e., ``last_from >= first_from``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``view_from``, ``view_dest``: views to read elements from ``view_from`` and write to ``view_dest``

  - must be rank-1, and have ``LayoutLeft``, ``LayoutRight``, or ``LayoutStride``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``bin_op``:

  - *binary* functor representing the operation to apply to each pair of elements. Must be valid to be called from the execution space passed, and callable with two arguments ``a,b`` of type (possible const) ``value_type``, where ``value_type`` is the value type of ``InputIteratorType`` (for 1,2,3,4) or the value type of ``view_from`` (for 5,6,7,8), and must not modify ``a,b``.

  - must conform to:

  .. code-block:: cpp

     struct BinaryOp
     {
       KOKKOS_INLINE_FUNCTION
       return_type operator()(const value_type & a,
                              const value_type & b) const {
         return /* ... */;
       }

       // or, also valid
       return_type operator()(value_type a,
                              value_type b) const {
         return /* ... */;
       }
     };

  The return type ``return_type`` must be such that an object of type ``OutputIteratorType`` for (1,2,3,4)
  or an object of type ``value_type`` where ``value_type`` is the value type of ``view_dest`` for (5,6,7,8)
  can be dereferenced and assigned a value of type ``return_type``.

Return Value
~~~~~~~~~~~~

Iterator to the element *after* the last element written.