# `ThreadVectorMDRange`

Header File: `Kokkos_Core.hpp`

Usage: 
  ```c++
    parallel_for(ThreadVectorMDRange<Kokkos::Rank<...>, TeamHandle>(team, extent1, extent2, ...),
      [=] (int i1, int i2, ...) {...});
    parallel_reduce(ThreadVectorMDRange<Kokkos::Rank<...>, TeamHandle>(team, extent1, extent2, ...), 
      [=] (int i1, int i2, ..., double& lsum) {...}, sum);
  ```

ThreadVectorMDRange is a [nested execution policy](https://kokkos.github.io/kokkos-core-wiki/ProgrammingGuide/HierarchicalParallelism.html?highlight=nested#nested-parallelism)
used inside of hierarchical parallelism. 

## Interface
  ```c++
   template <unsigned N, ..., typename TeamHandle>
   struct ThreadVectorMDRange<Rank<N, ...>, TeamHandle> {
     ThreadVectorMDRange(team, extent1, extent2, ..., extentN) { /* ... */ }

     /* ... */
   };
  ```

## Description

 * ```c++
   template <unsigned N, ..., typename TeamHandle>
   struct ThreadVectorMDRange<Rank<N, ...>, TeamHandle>;
   ```
   Splits the index range `0` to `extent` over the vector lanes of the calling thread,
   where extent is the backend dependent rank that will be vectorized

    *  **Arguments**
        * `team`: TeamHandle to the calling team execution context.
        * `extent_i`: index range length of each rank.

    * **Requirements**
        * `TeamHandle` is a type that models [TeamHandle](Kokkos%3A%3ATeamHandleConcept)
        * extents are ints.
        * This function can not be called inside a parallel operation dispatched using a
          `TeamVectorRange` policy, `TeamVectorRange` policy, `TeamVectorMDRange` policy
          or `ThreadVectorMDRange` policy.
        * `N >= 2 && N <= 8` is true;
  
## Examples

  ```c++
   using TeamHandle = TeamPolicy<>::member_type;

   parallel_for(TeamPolicy<>(N, Kokkos::AUTO),
     KOKKOS_LAMBDA(TeamHandle const& team) {
       int leagueRank = team.league_rank();

       auto teamThreadRange = TeamThreadRange(team, n0);
       auto threadVectorMDRange =
           ThreadVectorMDRange<Rank<3>, TeamHandle>(
               team, n1, n2, n3);

       parallel_for(teamThreadRange, [=](int i0) {
         parallel_for(threadVectorMDRange, [=](int i1, int i2, int i3) {
           A(leagueRank, i0, i1, i2, i3) += B(leagueRank, i1) + C(i1, i2, i3);
         });
       });

       team.team_barrier();

       int teamSum = 0;

       parallel_for(teamThreadRange, [=, &teamSum](int const& i0) {
         int threadSum = 0;
         parallel_reduce(threadVectorMDRange,
           [=](int i1, int i2, int i3, int& vectorSum) {
             vectorSum += D(leagueRank, i0, i1, i2, i3);
           }, threadSum
         );

         teamSum += threadSum;
       });
   });
  ```

