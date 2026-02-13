
``replace_if``
=================

Header: ``<Kokkos_StdAlgorithms.hpp>``

Description
-----------

Replaces all the elements in
the range ``[first, last)``  for which ``pred`` is ``true`` with with ``new_value``.
The overload taking a ``View`` uses the ``begin`` and ``end`` iterators of the ``View``.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.


Overload set accepting execution space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <class ExecutionSpace, class IteratorType, class UnaryPredicateType, class T>
   void replace_if(const ExecutionSpace& exespace,                              (1)
                   IteratorType first, IteratorType last,
                   UnaryPredicateType pred, const T& new_value);

   template <class ExecutionSpace, class IteratorType, class UnaryPredicateType, class T>
   void replace_if(const std::string& label, const ExecutionSpace& exespace,    (2)
                   IteratorType first, IteratorType last,
                   UnaryPredicateType pred, const T& new_value);

   template <class ExecutionSpace, class DataType, class... Properties, class UnaryPredicateType, class T>
   void replace_if(const ExecutionSpace& exespace,                              (3)
                   const Kokkos::View<DataType, Properties...>& view,
                   UnaryPredicateType pred, const T& new_value);

   template <class ExecutionSpace, class DataType, class... Properties, class UnaryPredicateType, class T>
   void replace_if(const std::string& label, const ExecutionSpace& exespace,    (4)
                   const Kokkos::View<DataType, Properties...>& view,
                   UnaryPredicateType pred, const T& new_value);

Overload set accepting a team handle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 4.2

.. code-block:: cpp

   template <class TeamHandleType, class InputIterator, class Predicate,
             class ValueType>
   KOKKOS_FUNCTION
   void replace_if(const TeamHandleType& teamHandle,                            (5)
                   InputIterator first, InputIterator last,
                   Predicate pred, const ValueType& new_value);

   template <class TeamHandleType, class DataType1, class... Properties1,
             class Predicate, class ValueType>
   KOKKOS_FUNCTION
   void replace_if(const TeamHandleType& teamHandle,                            (6)
                   const ::Kokkos::View<DataType1, Properties1...>& view,
                   Predicate pred, const ValueType& new_value);


Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``exespace``, ``first``, ``last``, ``view``, ``new_value``: same as in [``replace``](./StdReplace)

- ``teamHandle``: team handle instance given inside a parallel region when using a TeamPolicy

- ``label``: used to name the implementation kernels for debugging purposes

  - for 1, the default string is: "Kokkos::replace_if_iterator_api_default"

  - for 3, the default string is: "Kokkos::replace_if_view_api_default"

  - NOTE: overloads accepting a team handle do not use a label internally

- ``pred``: *unary* predicate returning ``true`` for the required element to replace.

  ``pred(v)`` must be valid to be called from the execution space passed, or
  the execution space associated with the team handle, and convertible 
  to bool for every argument ``v`` of type (possible const) ``value_type``, 
  where ``value_type`` is the value type of ``IteratorType`` or the value type 
  of ``view``, and must not modify ``v``.

  - must conform to:

  .. code-block:: cpp

   struct Predicate
   {
      KOKKOS_INLINE_FUNCTION
      bool operator()(const value_type & v) const { return /* ... */; }

      // or, also valid

      KOKKOS_INLINE_FUNCTION
      bool operator()(value_type v) const { return /* ... */; }
   };


Return Value
~~~~~~~~~~~~

None

Example
~~~~~~~~~~~~

.. code-block:: cpp

   template <class ValueType>
   struct IsPositiveFunctor {
     KOKKOS_INLINE_FUNCTION
     bool operator()(const ValueType val) const { return (val > 0); }
   };
   // ---

   namespace KE = Kokkos::Experimental;
   Kokkos::View<double*> a("a", 13);
   // do something with a
   // ...

   const double oldValue{2};
   const double newValue{34};
   KE::replace_if(Kokkos::DefaultExecutionSpace(), KE::begin(a), KE::end(a),
      IsPositiveFunctor<double>(), newValue);

   // explicitly set label and execution space (assuming active)
   KE::replace_if("mylabel", Kokkos::OpenMP(), a,
      IsPositiveFunctor<double>(), newValue);
