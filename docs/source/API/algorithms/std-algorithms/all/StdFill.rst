
``fill``
=========

Header: ``<Kokkos_StdAlgorithms.hpp>``

Description
-----------

Assigns a given ``value`` to each element in a given range or rank-1 ``View``.


Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.


Overload set accepting execution space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   template <class ExecutionSpace, class IteratorType, class T>
   void fill(const ExecutionSpace& exespace,                                    (1)
             IteratorType first, IteratorType last,
             const T& value);

   template <class ExecutionSpace, class IteratorType, class T>
   void fill(const std::string& label, const ExecutionSpace& exespace,          (2)
             IteratorType first, IteratorType last,
             const T& value);

   template <class ExecutionSpace, class DataType, class... Properties, class T>
   void fill(const ExecutionSpace& exespace,                                    (3)
             const Kokkos::View<DataType, Properties...>& view,
             const T& value);

   template <class ExecutionSpace, class DataType, class... Properties, class T>
   void fill(const std::string& label, const ExecutionSpace& exespace,          (4)
             const Kokkos::View<DataType, Properties...>& view,
             const T& value);

Overload set accepting a team handle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. versionadded:: 4.2

.. code-block:: cpp

   template <class TeamHandleType, class IteratorType, class T>
   KOKKOS_FUNCTION
   void fill(const TeamHandleType& teamHandle,                                  (5)
             IteratorType first, IteratorType last,
             const T& value);

   template <class TeamHandleType, class DataType, class... Properties, class T>
   KOKKOS_FUNCTION
   void fill(const TeamHandleType& teamHandle,                                  (6)
             const Kokkos::View<DataType, Properties...>& view,
             const T& value);


Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``exespace``: execution space instance

- ``teamHandle``: team handle instance given inside a parallel region when using a TeamPolicy

- ``label``: string forwarded to internal parallel kernels for debugging purposes

  - for 1, the default string is: "Kokkos::fill_iterator_api_default"

  - for 3, the default string is: "Kokkos::fill_view_api_default"

  - NOTE: overloads accepting a team handle do not use a label internally

- ``first, last``: range of elements to modify

  - must be *random access iterators*, e.g., ``Kokkos::Experimental::begin/end``

  - must represent a valid range, i.e., ``last >= first``

  - must be accessible from ``exespace`` or from the execution space associated with the team handle

- ``view``:

  - must be rank-1, and have ``LayoutLeft``, ``LayoutRight``, or ``LayoutStride``

  - must be accessible from ``exespace``

- ``value``: value to assign to each element


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
