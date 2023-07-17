``reduce``
==========

Header: ``Kokkos_StdAlgorithms.hpp``

Description
-----------

- Overload set A: performs a reduction of the elements in a range or in ``view``.

- Overload set B: performs a reduction of the elements in a range or in ``view`` accounting for the initial value ``init_reduction_value``.

- Overload set C: performs a reduction of the elements in a range or in ``view`` accounting for the initial value ``init_reduction_value`` using the functor  ``joiner`` to join operands during the reduction operation.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.

.. code-block:: cpp

   //
   // overload set A
   //
   template <class ExecutionSpace, class IteratorType>
   typename IteratorType::value_type reduce(const ExecutionSpace& exespace,      (1)
                                            IteratorType first,
                                            IteratorType last);

   template <class ExecutionSpace, class IteratorType>
   typename IteratorType::value_type reduce(const std::string& label,            (2)
                                            const ExecutionSpace& exespace,
                                            IteratorType first,
                                            IteratorType last);

   template <class TeamHandleType, class IteratorType>
   KOKKOS_FUNCTION
   typename IteratorType::value_type reduce(const TeamHandleType& teamHandle,    (3)
                                            IteratorType first,
                                            IteratorType last);

   template <class ExecutionSpace, class DataType, class... Properties>
   auto reduce(const ExecutionSpace& exespace,                                   (4)
               const Kokkos::View<DataType, Properties...>& view)

   template <class ExecutionSpace, class DataType, class... Properties>
   auto reduce(const std::string& label, const ExecutionSpace& exespace,         (5)
               const Kokkos::View<DataType, Properties...>& view);

   template <class TeamHandleType, class DataType, class... Properties>
   KOKKOS_FUNCTION
   auto reduce(const TeamHandleType& teamHandle,                                 (6)
               const Kokkos::View<DataType, Properties...>& view)

   //
   // overload set B
   //
   template <class ExecutionSpace, class IteratorType, class ValueType>
   ValueType reduce(const ExecutionSpace& exespace,                              (7)
                    IteratorType first, IteratorType last,
                    ValueType init_reduction_value);

   template <class ExecutionSpace, class IteratorType, class ValueType>
   ValueType reduce(const std::string& label,                                    (8)
                    const ExecutionSpace& exespace,
                    IteratorType first, IteratorType last,
                    ValueType init_reduction_value);

   template <class TeamHandleType, class IteratorType, class ValueType>
   KOKKOS_FUNCTION
   ValueType reduce(const TeamHandleType& teamHandle,                            (9)
                    IteratorType first, IteratorType last,
                    ValueType init_reduction_value);

   template <
      class ExecutionSpace, class DataType,
      class... Properties, class ValueType>
   ValueType reduce(const ExecutionSpace& exespace,                              (10)
                    const Kokkos::View<DataType, Properties...>& view,
                    ValueType init_reduction_value);

   template <
      class ExecutionSpace, class DataType,
      class... Properties, class ValueType>
   ValueType reduce(const std::string& label,                                    (11)
                    const ExecutionSpace& exespace,
                    const Kokkos::View<DataType, Properties...>& view,
                    ValueType init_reduction_value);

   template <
      class TeamHandleType, class DataType,
      class... Properties, class ValueType>
   KOKKOS_FUNCTION
   ValueType reduce(const TeamHandleType& teamHandle,                            (12)
                    const Kokkos::View<DataType, Properties...>& view,
                    ValueType init_reduction_value);

   //
   // overload set C
   //
   template <
      class ExecutionSpace, class IteratorType,
      class ValueType, class BinaryOp>
   ValueType reduce(const ExecutionSpace& exespace,                              (13)
                    IteratorType first, IteratorType last,
                    ValueType init_reduction_value,
                    BinaryOp joiner);

   template <
      class ExecutionSpace, class IteratorType,
      class ValueType, class BinaryOp>
   ValueType reduce(const std::string& label,                                    (14)
                    const ExecutionSpace& exespace,
                    IteratorType first, IteratorType last,
                    ValueType init_reduction_value,
                    BinaryOp joiner);

   template <
      class TeamHandleType, class IteratorType,
      class ValueType, class BinaryOp>
   KOKKOS_FUNCTION
   ValueType reduce(const TeamHandleType& teamHandle,                            (15)
                    IteratorType first, IteratorType last,
                    ValueType init_reduction_value,
                    BinaryOp joiner);

   template <
      class ExecutionSpace, class DataType,
      class... Properties, class ValueType, class BinaryOp>
   ValueType reduce(const ExecutionSpace& exespace,                              (16)
                    const Kokkos::View<DataType, Properties...>& view,
                    ValueType init_reduction_value,
                    BinaryOp joiner);

   template <
      class ExecutionSpace, class DataType,
      class... Properties, class ValueType, class BinaryOp>
   ValueType reduce(const std::string& label,                                    (17)
                    const ExecutionSpace& exespace,
                    const Kokkos::View<DataType, Properties...>& view,
                    ValueType init_reduction_value,
                    BinaryOp joiner);

   template <
      class TeamHandleType, class DataType,
      class... Properties, class ValueType, class BinaryOp>
   KOKKOS_FUNCTION
   ValueType reduce(const TeamHandleType& teamHandle,                            (18)
                    const Kokkos::View<DataType, Properties...>& view,
                    ValueType init_reduction_value,
                    BinaryOp joiner);


Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``exespace``: execution space instance

- ``teamHandle``: team handle instance given inside a parallel region when using a TeamPolicy

- ``label``: string forwarded to internal parallel kernels for debugging purposes

  - 1,7,13: The default string is "Kokkos::reduce_default_functors_iterator_api"

  - 3,10: The default string is "Kokkos::reduce_default_functors_view_api"

  - 16: The default string is "Kokkos::reduce_custom_functors_view_api"

  - NOTE: overloads accepting a team handle do not use a label internally

- ``first``, ``last``: range of elements to reduce over

  - must be *random access iterators*, e.g., returned from ``Kokkos::Experimental::(c)begin/(c)end``

  - must represent a valid range, i.e., ``last_from >= first_from``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``view``: view to reduce

  - must be rank-1, and have ``LayoutLeft``, ``LayoutRight``, or ``LayoutStride``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``init_reduction_value``: initial reduction value to use

- ``joiner``:

  - *binary* functor performing the desired operation to join two elements. Must be valid to be called from the execution space passed, and callable with two arguments ``a,b`` of type (possible const) ``ValueType``, and must not modify ``a,b``.

  - Must conform to:

  .. code-block:: cpp

     struct JoinFunctor {
	    KOKKOS_FUNCTION
	    constexpr ValueType operator()(const ValueType& a, const ValueType& b) const {
	      return /* ... */
	    }
     };

  - The behavior is non-deterministic if the ``joiner`` operation is not associative or not commutative.

Return Value
~~~~~~~~~~~~

The reduction result.
