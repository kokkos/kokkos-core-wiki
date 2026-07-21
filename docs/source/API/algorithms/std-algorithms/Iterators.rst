Iterators
=========

.. role:: cpp(code)
    :language: cpp

``Kokkos::Experimental::{begin, cbegin, end, cend}``
----------------------------------------------------

Header File: ``<Kokkos_StdAlgorithms.hpp>`` (until Kokkos 5.2), ``<Kokkos_Core.hpp>`` (since Kokkos 5.2)

See `Iterators (Core) <../../core/view/iterators.html>`__.

------------------

``Kokkos::Experimental::distance``
----------------------------------

Header File: ``<Kokkos_StdAlgorithms.hpp>`` (until Kokkos 5.2), ``<Kokkos_Core.hpp>`` (since Kokkos 5.2)

See `Iterators (Core) <../../core/view/iterators.html>`__.

------------------

``Kokkos::Experimental::iter_swap``
-----------------------------------

Header File: ``<Kokkos_StdAlgorithms.hpp>``

.. cpp:function:: template <class IteratorType> void iter_swap(IteratorType first, IteratorType last);

   Swaps the values of the elements the given iterators are pointing to.

Parameters and Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ``first, last``: iterators to swap

Notes
~~~~~

Currently, the API does not have an execution space parameter because the operation is performed in the *default execution space*. The operation fences the default execution space.

Return
~~~~~~

None

Example
~~~~~~~

.. code-block:: cpp

    namespace KE = Kokkos::Experimental;
    Kokkos::View<double*> a("a", 13);

    auto it1 = KE::begin(a);
    auto it2 = it1 + 4;
    KE::swap(it1, it2);
