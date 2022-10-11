# `Kokkos::TeamVectorMDRange`

Header File: `Kokkos_Core.hpp`

Usage: 
  ```c++
    parallel_for(TeamVectorMDRange<Kokkos::Rank<...>, TeamHandle>(team, extent1, extent2, ...),
      [=] (int i1, int i2, ...) {...});
    parallel_reduce(TeamVectorMDRange<Kokkos::Rank<...>, TeamHandle>(team, extent1, extent2, ...), 
      [=] (int i1, int i2, ..., double& lsum) {...}, sum);
  ```

TeamVectorMDRange is a [nested execution policy](https://kokkos.github.io/kokkos-core-wiki/ProgrammingGuide/HierarchicalParallelism.html?highlight=nested#nested-parallelism)
used inside of hierarchical parallelism. 

## Interface
  ```c++
   template <unsigned N, ..., typename TeamHandle>
   struct TeamVectorMDRange<Rank<N, ...>, TeamHandle> {
     TeamVectorMDRange(team, extent1, extent2, ..., extentN) { /* ... */ }

     /* ... */
   };
  ```

## Description

 * ```c++
   template <unsigned N, ..., typename TeamHandle>
   struct TeamVectorMDRange<Rank<N, ...>, TeamHandle>;
   ```
   Splits an index range `0` to `extent1` over the threads of the team and
   another index range `0` to `extent2` over their vector lanes.
   Ranks for threading and vectorization determined by the backend.

    *  **Arguments**
        * `team`: TeamHandle to the calling team execution context.
        * `extent_i`: index range length of each rank.

    *  **Requirements**
        * `TeamHandle` is a type that models [TeamHandle](Kokkos%3A%3ATeamHandleConcept)
        * extents are ints.
        * Every member thread of `team` must call the operation in the same branch, i.e. it is not legal to have some 
          threads call this function in one branch, and the other threads of `team` call it in another branch.
        * `N >= 2 && N <= 8` is true;
 
## Examples

  ```c++
   typedef TeamPolicy<>::member_type team_handle;
   parallel_for(TeamPolicy<>(N,AUTO,4), KOKKOS_LAMBDA (const team_handle& team) {
     int n = team.league_rank();
     parallel_for(TeamVectorRange(team,M), [&] (const int& i) {
       A(n,i) = B(n) + i;
     });
     team.team_barrier();
     int team_sum;
     parallel_reduce(TeamVectorRange(team,M), [&] (const int& i, int& lsum) {
       lsum += A(n,i);
     },team_sum);
     single(PerTeam(team),[&] () {
       A_rowsum(n) += team_sum;
     });
   });

   using TeamHandle = TeamPolicy<>::member_type;

   parallel_for(TeamPolicy<>(N,AUTO),
     KOKKOS_LAMBDA(TeamHandle const& team) {

       int leagueRank = team.league_rank();

       auto teamVectorMDRange =
           TeamVectorMDRange<Rank<4>, TeamType>(
               team, n0, n1, n2, n3);

       parallel_for(teamVectorMDRange
         [=](int i0, int i1, int i2, int i3) {
           A(leagueRank,i0) = B(leagueRank, i1) + C(i1, i2, i3);
       });

       team.team_barrier();

       int teamSum = 0;

       parallel_reduce(teamVectorMDRange,
           [=](int i0, int i1, int i2, int i3, int& vectorSum) {
             vectorSum += v(leagueRank, i, j, k, l);
           }, teamSum
       );

       single(PerTeam(team), [&leagueSum, teamSum]() { leagueSum += teamSum; });

       A_rowSum[leagueRank] = leagueSum;
     });

 
  ```

