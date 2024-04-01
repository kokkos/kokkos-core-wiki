``transform_reduce``
====================

Header: ``Kokkos_StdAlgorithms.hpp``

Description
-----------

Performs the *product* (via ``operator*``, the functor ``binary_transformer`` or the functor ``unary_transformer``) between each pair of elements in:

- the range ``[first1, last1)`` and the range starting at ``first2``, or

- ``first_view`` and ``second_view``, or

- the range ``[first, last)``,  or

- ``view``,

and reduces the results along with the initial value ``init_reduction_value`` and with the join operation done via the *binary* functor ``joiner`` if needed.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.

Overload set accepting execution space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <
      class ExecutionSpace, class IteratorType1,
      class IteratorType2, class ValueType>
   ValueType transform_reduce(const ExecutionSpace& exespace,                             (1)
                              IteratorType1 first1, IteratorType1 last1,
                              IteratorType2 first2,
                              ValueType init_reduction_value);

   template <
      class ExecutionSpace, class IteratorType1,
      class IteratorType2, class ValueType>
   ValueType transform_reduce(const std::string& label,                                   (2)
                              const ExecutionSpace& exespace,
                              IteratorType1 first1, IteratorType1 last1,
                              IteratorType2 first2,
                              ValueType init_reduction_value);

   template <
      class ExecutionSpace,
      class DataType1, class... Properties1,
      class DataType2, class... Properties2,
      class ValueType>
   ValueType transform_reduce(const ExecutionSpace& exespace,                             (3)
                              const Kokkos::View<DataType1, Properties1...>& first_view,
                              const Kokkos::View<DataType2, Properties2...>& second_view,
                              ValueType init_reduction_value);

   template <
      class ExecutionSpace,
      class DataType1, class... Properties1,
      class DataType2, class... Properties2,
      class ValueType>
   ValueType transform_reduce(const std::string& label,                                   (4)
                              const ExecutionSpace& exespace,
                              const Kokkos::View<DataType1, Properties1...>& first_view,
                              const Kokkos::View<DataType2, Properties2...>& second_view,
                              ValueType init_reduction_value);

   template <
      class ExecutionSpace,
      class IteratorType1, class IteratorType2,
      class ValueType,
      class BinaryJoinerType, class BinaryTransform>
   ValueType transform_reduce(const ExecutionSpace& exespace,                             (5)
                              IteratorType1 first1, IteratorType1 last1,
                              IteratorType2 first2,
                              ValueType init_reduction_value,
                              BinaryJoinerType joiner,
                              BinaryTransform binary_transformer);

   template <
      class ExecutionSpace,
      class IteratorType1, class IteratorType2,
      class ValueType,
      class BinaryJoinerType, class BinaryTransform>
   ValueType transform_reduce(const std::string& label,                                   (6)
                              const ExecutionSpace& exespace,
                              IteratorType1 first1, IteratorType1 last1,
                              IteratorType2 first2,
                              ValueType init_reduction_value,
                              BinaryJoinerType joiner,
                              BinaryTransform binary_transformer);

   template <
      class ExecutionSpace,
      class DataType1, class... Properties1,
      class DataType2, class... Properties2,
      class ValueType,
      class BinaryJoinerType, class BinaryTransform>
   ValueType transform_reduce(const ExecutionSpace& exespace,                             (7)
                              const Kokkos::View<DataType1, Properties1...>& first_view,
                              const Kokkos::View<DataType2, Properties2...>& second_view,
                              ValueType init_reduction_value,
                              BinaryJoinerType joiner,
                              BinaryTransform binary_transformer);

   template <
      class ExecutionSpace,
      class DataType1, class... Properties1,
      class DataType2, class... Properties2,
      class ValueType,
      class BinaryJoinerType, class BinaryTransform>
   ValueType transform_reduce(const std::string& label,                                   (8)
                              const ExecutionSpace& exespace,
                              const Kokkos::View<DataType1, Properties1...>& first_view,
                              const Kokkos::View<DataType2, Properties2...>& second_view,
                              ValueType init_reduction_value,
                              BinaryJoinerType joiner,
                              BinaryTransform binary_transformer);

   template <
      class ExecutionSpace,
      class IteratorType, class ValueType,
      class BinaryJoinerType, class UnaryTransform>
   ValueType transform_reduce(const ExecutionSpace& exespace,                             (9)
                              IteratorType first1, IteratorType last1,
                              ValueType init_reduction_value,
                              BinaryJoinerType joiner,
                              UnaryTransform unary_transformer);

   template <
      class ExecutionSpace,
      class IteratorType, class ValueType,
      class BinaryJoinerType, class UnaryTransform>
   ValueType transform_reduce(const std::string& label,                                   (10)
                              const ExecutionSpace& exespace,
                              IteratorType first1, IteratorType last1,
                              ValueType init_reduction_value,
                              BinaryJoinerType joiner,
                              UnaryTransform unary_transformer);

   template <
      class ExecutionSpace,
      class DataType, class... Properties, class ValueType,
      class BinaryJoinerType, class UnaryTransform>
   ValueType transform_reduce(const ExecutionSpace& exespace,                             (11)
                              const Kokkos::View<DataType, Properties...>& view,
                              ValueType init_reduction_value,
                              BinaryJoinerType joiner,
                              UnaryTransform unary_transformer);

   template <
      class ExecutionSpace,
      class DataType, class... Properties, class ValueType,
      class BinaryJoinerType, class UnaryTransform>
   ValueType transform_reduce(const std::string& label,                                   (12)
                              const ExecutionSpace& exespace,
                              const Kokkos::View<DataType, Properties...>& view,
                              ValueType init_reduction_value,
                              BinaryJoinerType joiner,
                              UnaryTransform unary_transformer);


Overload set accepting a team handle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 4.2

.. code-block:: cpp

   template <
      class TeamHandleType,
      class IteratorType1, class IteratorType2,
      class ValueType>
   KOKKOS_FUNCTION
   ValueType transform_reduce(const TeamHandleType& teamHandle,                           (13)
                              IteratorType1 first1, IteratorType1 last1,
                              IteratorType2 first2,
                              ValueType init_reduction_value);

   template <
      class TeamHandleType,
      class DataType1, class... Properties1,
      class DataType2, class... Properties2,
      class ValueType>
   KOKKOS_FUNCTION
   ValueType transform_reduce(const TeamHandleType& teamHandle,                           (14)
                              const Kokkos::View<DataType1, Properties1...>& first_view,
                              const Kokkos::View<DataType2, Properties2...>& second_view,
                              ValueType init_reduction_value);

   template <
      class TeamHandleType,
      class IteratorType1, class IteratorType2,
      class ValueType,
      class BinaryJoinerType, class BinaryTransform>
   KOKKOS_FUNCTION
   ValueType transform_reduce(const TeamHandleType& teamHandle,                           (15)
                              IteratorType1 first1, IteratorType1 last1,
                              IteratorType2 first2,
                              ValueType init_reduction_value,
                              BinaryJoinerType joiner,
                              BinaryTransform binary_transformer);

   template <
      class TeamHandleType,
      class DataType1, class... Properties1,
      class DataType2, class... Properties2,
      class ValueType,
      class BinaryJoinerType, class BinaryTransform>
   KOKKOS_FUNCTION
   ValueType transform_reduce(const TeamHandleType& teamHandle,                           (16)
                              const Kokkos::View<DataType1, Properties1...>& first_view,
                              const Kokkos::View<DataType2, Properties2...>& second_view,
                              ValueType init_reduction_value,
                              BinaryJoinerType joiner,
                              BinaryTransform binary_transformer);

   template <
      class TeamHandleType,
      class IteratorType, class ValueType,
      class BinaryJoinerType, class UnaryTransform>
   KOKKOS_FUNCTION
   ValueType transform_reduce(const TeamHandleType& teamHandle,                           (17)
                              IteratorType first1, IteratorType last1,
                              ValueType init_reduction_value,
                              BinaryJoinerType joiner,
                              UnaryTransform unary_transformer);

   template <
      class TeamHandleType,
      class DataType, class... Properties,
      class ValueType,
      class BinaryJoinerType, class UnaryTransform>
   KOKKOS_FUNCTION
   ValueType transform_reduce(const TeamHandleType& teamHandle,                           (18)
                              const Kokkos::View<DataType, Properties...>& view,
                              ValueType init_reduction_value,
                              BinaryJoinerType joiner,
                              UnaryTransform unary_transformer);

Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``exespace``: execution space instance

- ``teamHandle``: team handle instance given inside a parallel region when using a TeamPolicy

- ``label``: string forwarded to internal parallel kernels for debugging purposes

  - 1, 3: The default string is "Kokkos::transform_reduce_default_functors_iterator_api"

  - 7, 13: The default string is "Kokkos::transform_reduce_custom_functors_iterator_api"

  - 9, 15: The default string is "Kokkos::transform_reduce_custom_functors_view_api"

  - NOTE: overloads accepting a team handle do not use a label internally

- ``first1``, ``last1``, ``first2``: ranges of elements to transform and reduce

  - must be *random access iterators*, e.g., returned from ``Kokkos::Experimental::(c)begin/(c)end``

  - must represent a valid range, i.e., ``last_from >= first_from``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``first_view``, ``second_view``: views to transform and reduce

  - must be rank-1, and have ``LayoutLeft``, ``LayoutRight``, or ``LayoutStride``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``init_reduction_value``: initial reduction value to use

- ``joiner``:

  - *binary* functor performing the desired operation to join two elements. Must be valid to be called from the execution space passed, and callable with two arguments ``a,b`` of type (possible const) ``ValueType``, and must not modify ``a,b``.

  - Must conform to:

  .. code-block:: cpp

     struct JoinFunctor {
	    KOKKOS_FUNCTION
	    constexpr ValueType operator()(const ValueType& a,
                                      const ValueType& b) const {
	      return /* ... */
	    }
     };

  - The behavior is non-deterministic if the ``joiner`` operation is not associative or not commutative.

- ``binary_transformer``:

  - *binary* functor applied to each pair of elements *before* doing the reduction. Must be valid to be called from the execution space passed, and callable with two arguments ``a,b`` of type (possible const) ``value_type_a`` and ``value_type_b``, where ``value_type_{a,b}`` are the value types of ``first1`` and ``first2`` or the value types of ``first_view`` and ``second_view``, and must not modify ``a,b``.

  - Must conform to:

  .. code-block:: cpp

     struct BinaryTransformer {
       KOKKOS_FUNCTION
       constexpr return_type operator()(const value_type_a & a, const value_type_b & b) const {
         return /* ... */
       }
     };

  - the ``return_type`` is such that it can be accepted by the ``joiner``

- ``unary_transformer``:

  - *unary* functor performing the desired operation to an element. Must be valid to be called from the execution space passed, and callable with an arguments ``v`` of type (possible const) ``value_type``, where ``value_type`` is the value type of ``first1`` or the value type of ``first_view``, and must not modify ``v``.

  - Must conform to:

  .. code-block:: cpp

     struct UnaryTransformer {
       KOKKOS_FUNCTION
       constexpr value_type operator()(const value_type & v) const {
         return /* ... */
       }
     };

Return Value
~~~~~~~~~~~~

The reduction result.
