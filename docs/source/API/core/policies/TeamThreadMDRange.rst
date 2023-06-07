``TeamThreadMDRange``
=====================

.. role::cpp(code)
    :language: cpp

Header File: ``<Kokkos_Core.hpp>``

Usage:
------

.. code-block:: cpp

   parallel_for(TeamThreadMDRange<Kokkos::Rank<...>, TeamHandle>(team, extent1, extent2, ...),
     [=] (int i1, int i2, ...) {...});
   parallel_reduce(TeamThreadMDRange<Kokkos::Rank<...>, TeamHandle>(team, extent1, extent2, ...),
     [=] (int i1, int i2, ..., double& lsum) {...}, sum);


TeamThreadMDRange is a a `nested execution policy <./NestedPolicies.html>`_  used inside of hierarchical parallelism.

Interface
---------

.. code-block:: cpp

   template <unsigned N, ..., typename TeamHandle>
   struct TeamThreadMDRange<Rank<N, ...>, TeamHandle>
   {
     TeamThreadMDRange(team, extent1, extent2, ..., extentN) { /* ... */ }
   };

Splits the index range ``0`` to ``extent`` over the threads of the team,
where extent is the backend dependent rank that will be threaded

*  **Arguments**

   * ``team``: TeamHandle to the calling team execution context.

   * ``extent_i``: index range length of each rank.

* **Requirements**

  * ``TeamHandle`` is a type that models `TeamHandle <./TeamHandleConcept.html>`_

  * extents are ints.

  * Every member thread of ``team`` must call the operation in the same branch, i.e. it is not legal to have some
    threads call this function in one branch, and the other threads of ``team`` call it in another branch.

  * ``N >= 2 && N <= 8`` is true;


Examples
--------

.. code-block:: cpp

   using TeamHandle = TeamPolicy<>::member_type;

   parallel_for(TeamPolicy<>(N,AUTO),
     KOKKOS_LAMBDA (TeamHandle const& team) {

       int leagueRank = team.league_rank();

       auto teamThreadMDRange =
           TeamThreadMDRange<Rank<4>, TeamHandle>(
               team, n0, n1, n2, n3);

       parallel_for(teamThreadMDRange, [=](int i0, int i1, int i2, int i3) {
         A(leagueRank, i0, i1, i2, i3) = B(leagueRank, i1) + C(i1, i2, i3);
       });

       team.team_barrier();

       int teamSum = 0;

       parallel_reduce(teamThreadMDRange,
         [=](int i0, int i1, int i2, int i3, int& threadSum) {
           threadSum += D(leagueRank, i0, i1, i2, i3);
         }, teamSum
       );

       single(PerTeam(team), [&leagueSum, teamSum]() { leagueSum += teamSum; });

       A_rowSum[leagueRank] = leagueSum;
   });
