
``replace``
============

Header: ``<Kokkos_StdAlgorithms.hpp>``

Description
-----------

Replaces with ``new_value`` all elements that are equal to ``old_value`` in 
a given range or rank-1 ``View``. Equality is checked using ``operator==``.

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.


Overload set accepting execution space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <class ExecutionSpace, class IteratorType, class T>
   void replace(const ExecutionSpace& exespace,                                 (1)
                IteratorType first, IteratorType last,
                const T& old_value, const T& new_value);

   template <class ExecutionSpace, class IteratorType, class T>
   void replace(const std::string& label, const ExecutionSpace& exespace,       (2)
                IteratorType first, IteratorType last,
                const T& old_value, const T& new_value);

   template <class ExecutionSpace, class DataType, class... Properties, class T>
   void replace(const ExecutionSpace& exespace,                                 (3)
                const Kokkos::View<DataType, Properties...>& view,
                const T& old_value, const T& new_value);

   template <class ExecutionSpace, class DataType, class... Properties, class T>
   void replace(const std::string& label, const ExecutionSpace& exespace,       (4)
                const Kokkos::View<DataType, Properties...>& view,
                const T& old_value, const T& new_value);


Overload set accepting a team handle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 4.2

.. code-block:: cpp

   template <class TeamHandleType, class Iterator, class ValueType>
   KOKKOS_FUNCTION
   void replace(const TeamHandleType& teamHandle,                               (5)
                Iterator first, Iterator last,
                const ValueType& old_value, const ValueType& new_value);

   template <class TeamHandleType, class DataType1, class... Properties1,
             class ValueType>
   KOKKOS_FUNCTION
   void replace(const TeamHandleType& teamHandle,                               (6)
                const ::Kokkos::View<DataType1, Properties1...>& view,
                const ValueType& old_value, const ValueType& new_value);


Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``exespace``: execution space instance

- ``teamHandle``: team handle instance given inside a parallel region when using a TeamPolicy

- ``label``: used to name the implementation kernels for debugging purposes

  - for 1, the default string is: "Kokkos::replace_iterator_api_default"

  - for 3, the default string is: "Kokkos::replace_view_api_default"

  - NOTE: overloads accepting a team handle do not use a label internally

- ``first, last``: range of elements to search in

  - must be *random access iterators*, e.g., ``Kokkos::Experimental::begin/end``

  - must represent a valid range, i.e., ``last >= first`` (this condition is checked in debug mode)

  - must be accessible from ``exespace``

- ``view``:

  - must be rank-1, and have ``LayoutLeft``, ``LayoutRight``, or ``LayoutStride``

  - must be accessible from ``exespace``

- ``old_value``, ``new_value``: self-explanatory


Return Value
~~~~~~~~~~~~

None

Example
~~~~~~~~~~~~

.. code-block:: cpp

   namespace KE = Kokkos::Experimental;
   Kokkos::View<double*> a("a", 13);

   KE::fill(Kokkos::DefaultExecutionSpace(), KE::begin(a), KE::end(a), 4.);

   // passing the view directly
   KE::fill(Kokkos::DefaultExecutionSpace(), a, 22.);

   // explicitly set execution space (assuming active)
   KE::fill(Kokkos::OpenMP(), KE::begin(a), KE::end(a), 14.);
