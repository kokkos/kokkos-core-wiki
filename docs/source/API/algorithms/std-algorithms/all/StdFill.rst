
``fill``
=========

Header: ``<Kokkos_StdAlgorithms.hpp>``

Description
-----------

Copy-assigns ``value`` to each element in the range ``[first, last)`` (overloads 1,2)
or in ``view`` (overloads 3,4).

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.


.. code-block:: cpp

   //
   // overload set accepting execution space
   //
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

   //
   // overload set accepting a team handle
   // Note: for now omit the overloads accepting a label
   // since they cause issues on device because of the string allocation.
   //
   template <class TeamHandleType, class IteratorType, class T>
   KOKKOS_FUNCTION
   void fill(const TeamHandleType& th,
             IteratorType first, IteratorType last,
             const T& value);

   template <class TeamHandleType, class DataType, class... Properties, class T>
   KOKKOS_FUNCTION
   void fill(const TeamHandleType& th,
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

- ``first, last``: range of elements to search in

  - range of elements to assign to

  - must be *random access iterators*, e.g., ``Kokkos::Experimental::begin/end``

  - must represent a valid range, i.e., ``last >= first`` (checked in debug mode)

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
