``AUTO``, ``AUTO_t``
====================

Defined in header ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

   Kokkos::TeamPolicy<>(league_size, Kokkos::AUTO);
   Kokkos::TeamPolicy<>(league_size, Kokkos::AUTO());


:cpp:var:`AUTO` is a tag used in place of the ``team_size`` argument of
:doc:`TeamPolicy` to let Kokkos determine, at launch time, a good team size
for the given functor and execution space.

Both ``Kokkos::AUTO`` and ``Kokkos::AUTO()`` syntax are supported.

Interface
---------

.. cpp:struct:: AUTO_t

   Tag type passed in place of ``team_size`` to :doc:`TeamPolicy` to
   request that Kokkos selects the team size automatically.

   .. cpp:function:: KOKKOS_FUNCTION constexpr const AUTO_t& operator()() const;

      Enables the ``Kokkos::AUTO()`` syntax.

      :returns: ``*this``


.. cpp:var:: inline constexpr AUTO_t AUTO{};

   Constant instance of :cpp:struct:`AUTO_t` used to request automatic
   team size selection for :doc:`TeamPolicy`.


Example
-------

.. code-block:: cpp

   // Let Kokkos pick a team size for N leagues
   Kokkos::parallel_for(Kokkos::TeamPolicy<>(N, Kokkos::AUTO),
     KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
       // ...
     });

   // Both syntaxes work
   Kokkos::TeamPolicy<> policy1(N, Kokkos::AUTO);
   Kokkos::TeamPolicy<> policy2(N, Kokkos::AUTO());

   // AUTO can be combined with an explicit vector length
   Kokkos::TeamPolicy<> policy3(N, Kokkos::AUTO, /*vector_length=*/8);


See Also
--------

.. seealso::

   :doc:`TeamPolicy`
      Execution policy for hierarchical (thread team) parallelism
