
``fill_n``
===========

Header: ``<Kokkos_StdAlgorithms.hpp>``

Description
-----------

Copy-assigns ``value`` to the first ``n`` elements in the range starting at ``first`` (overloads 1,2)
or the first ``n`` elements in ``view`` (overloads 3,4).

Interface
---------

.. warning:: This is currently inside the ``Kokkos::Experimental`` namespace.


.. code-block:: cpp

   //
   // overload set accepting execution space
   //
   template <class ExecutionSpace, class IteratorType, class SizeType, class T>
   IteratorType fill_n(const ExecutionSpace& exespace,                             (1)
                       IteratorType first,
                       SizeType n, const T& value);

   template <class ExecutionSpace, class IteratorType, class SizeType, class T>
   IteratorType fill_n(const std::string& label, const ExecutionSpace& exespace,   (2)
                       IteratorType first,
                       SizeType n, const T& value);

   template <class ExecutionSpace, class DataType, class... Properties, class SizeType, class T>
   auto fill_n(const ExecutionSpace& exespace,                                     (3)
               const Kokkos::View<DataType, Properties...>& view,
               SizeType n, const T& value);

   template <class ExecutionSpace, class DataType, class... Properties, class SizeType, class T>
   auto fill_n(const std::string& label, const ExecutionSpace& exespace,           (4)
               const Kokkos::View<DataType, Properties...>& view,
               SizeType n, const T& value);

   //
   // overload set accepting a team handle
   //
   template <class TeamHandleType, class IteratorType, class SizeType, class T>
   KOKKOS_FUNCTION
   IteratorType fill_n(const TeamHandleType& th,
                       IteratorType first, SizeType n,
                       const T& value);

   template <
       class TeamHandleType, class DataType, class... Properties, class SizeType,
       class T, int>
   KOKKOS_FUNCTION
   IteratorType fill_n(const TeamHandleType& th,
                       const Kokkos::View<DataType, Properties...>& view,
                       SizeType n,
                       const T& value);


Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``exespace``, ``first``, ``view``, ``value``: same as in [``fill``](./StdFill): execution space instance

- ``teamHandle``: team handle instance given inside a parallel region when using a TeamPolicy

- ``label``: used to name the implementation kernels for debugging purposes

  - for 1, the default string is: "Kokkos::fill_n_iterator_api_default"

  - for 3, the default string is: "Kokkos::fill_n_view_api_default"

  - NOTE: overloads accepting a team handle do not use a label internally

- ``n``: number of elements to modify (must be non-negative)

- ``value``: value to assign to each element

Return Value
~~~~~~~~~~~~

If ``n > 0``, returns an iterator to the element *after* the last element assigned.

Otherwise, it returns ``first`` (for 1,2) or ``Kokkos::begin(view)`` (for 3,4).


Example
~~~~~~~~~~~~

.. code-block:: cpp

   namespace KE = Kokkos::Experimental;
   Kokkos::View<double*> a("a", 13);
   // do something with a
   // ...

   const double newValue{4};
   KE::fill_n(Kokkos::DefaultExecutionSpace(), KE::begin(a), 10, newValue);

   // passing the view directly
   KE::fill_n(Kokkos::DefaultExecutionSpace(), a, 10, newValue);

   // explicitly set execution space (assuming active)
   KE::fill_n(Kokkos::OpenMP(), KE::begin(a), 10, newValue);
