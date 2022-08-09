# `Kokkos::MDTeamThreadRange`

Header File: `Kokkos_Core.hpp`

Usage: 
  ```c++
    parallel_for(MDTeamThreadRange<Kokkos::Rank<...>, TeamHandle>(team, extent1, extent2, ...),
      [=] (int i1, int i2, ...) {...});
    parallel_reduce(MDTeamThreadRange<Kokkos::Rank<...>, TeamHandle>(team, extent1, extent2, ...), 
      [=] (int i1, int i2, ..., double& lsum) {...}, sum);
  ```

MDTeamThreadRange is a [nested execution policy](https://kokkos.github.io/kokkos-core-wiki/ProgrammingGuide/HierarchicalParallelism.html?highlight=nested#nested-parallelism)
used inside of hierarchical parallelism.

## Interface
  ```c++
   template <unsigned N, ..., typename TeamHandle>
   struct MDTeamThreadRange<Rank<N, ...>, TeamHandle> {
     MDTeamThreadRange(team, extent1, extent2, ..., extentN) { /* ... */ }

     /* ... */
   };
  ```

## Description

 * ```c++
   template <unsigned N, ..., typename TeamHandle>
   struct MDTeamThreadRange<Rank<N, ...>, TeamHandle>;
   ```
   Splits the index range `0` to `extent` over the threads of the team,
   where extent is the backend dependent rank that will be threaded

    *  **Arguments**
        * `team`: TeamHandle to the calling team execution context.
        * `extent_i`: index range length of each rank.

    * **Requirements**
        * `TeamHandle` is a type that models [TeamHandle](Kokkos%3A%3ATeamHandleConcept)
        * extents are ints.
        * Every member thread of `team` must call the operation in the same branch, i.e. it is not legal to have some
          threads call this function in one branch, and the other threads of `team` call it in another branch.
        * `N >= 2 && N <= 8` is true;

  
## Examples

  ```c++
   using TeamHandle = TeamPolicy<>::member_type;

   parallel_for(TeamPolicy<>(N,AUTO),
     KOKKOS_LAMBDA (TeamHandle const& team) {

       int leagueRank = team.league_rank();

       auto mdTeamThreadRange =
           MDTeamThreadRange<Rank<4>, TeamHandle>(
               team, n0, n1, n2, n3);

       parallel_for(mdTeamThreadRange, [=](int i0, int i1, int i2, int i3) {
         A(leagueRank,i0) = B(leagueRank, i1) + C(i1, i2, i3);
       });

       team.team_barrier();

       int teamSum = 0;

       parallel_reduce(mdTeamThreadRange,
         [=](int i0, int i1, int i2, int i3, int& threadSum) {
           threadSum += D(leagueRank, i0, i1, i2, i3);
         }, teamSum
       );

       single(PerTeam(team), [&leagueSum, teamSum]() { leagueSum += teamSum; });

       A_rowSum[leagueRank] = leagueSum;
   });
  ```

