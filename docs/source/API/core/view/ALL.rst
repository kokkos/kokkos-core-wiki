``ALL``, ``ALL_t``
==================

Defined in header ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

   Kokkos::subview(v, i, Kokkos::ALL);
   Kokkos::subview(v, i, Kokkos::ALL());


:cpp:var:`ALL` is a slice specifier used with :cpp:func:`subview` to select all
elements along a dimension.

Both ``Kokkos::ALL`` and ``Kokkos::ALL()`` syntax are supported.

Interface
---------

.. cpp:struct:: ALL_t
   
   Slice specifier type that can be used with :cpp:func:`subview` to indicate
   the full range of indices in a dimension.

   .. cpp:function:: KOKKOS_FUNCTION constexpr const ALL_t& operator()() const;
   
      Enables the ``Kokkos::ALL()`` syntax.

      :returns: ``*this``

   .. cpp:function:: KOKKOS_FUNCTION constexpr bool operator==(const ALL_t&) const;
      
      Equality comparison operator.

      :returns: ``true``

      

.. cpp:var:: inline constexpr ALL_t ALL{};
   
   Constant instance of :cpp:struct:`ALL_t` used to select all indices in a
   dimension.

Example
-------

.. code-block:: cpp

   Kokkos::View<double**[5]> a("A", N0, N1);

   // Select all elements in dimensions 1 and 2, fix dimension 0 to index 5
   auto s = Kokkos::subview(a, 5, Kokkos::ALL, Kokkos::ALL);
   // Result: s has type View<double[5]> with dimensions (N1, 5)

   // Both syntaxes work
   auto s1 = Kokkos::subview(a, 5, Kokkos::ALL,   Kokkos::ALL);
   auto s2 = Kokkos::subview(a, 5, Kokkos::ALL(), Kokkos::ALL());

See Also
--------

.. seealso::

   :doc:`../view/subview`
      Create subviews of views

   :doc:`../stl-compat/pair`
      Specify a contiguous range of indices
