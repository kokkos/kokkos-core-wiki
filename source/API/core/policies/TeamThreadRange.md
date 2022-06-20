# `TeamThreadRange`

Header File: `Kokkos_Core.hpp`

Usage: 
```c++
parallel_for(TeamThreadRange(team,range), [=] (int i) {...});
parallel_reduce(TeamThreadRange(team,begin,end), 
  [=] (int i, double& lsum) {...},sum);
```

TeamThreadRange is a [nested execution policy](https://github.com/kokkos/kokkos/wiki/Execution-Policies#nested-execution-policies) used inside of hierarchical parallelism. 
In contrast to global policies, the public interface for nested policies is implemented 
as functions, in order to enable implicit templating on the execution space type via 
the team handle.

## Synopsis 
```c++
template<class TeamMemberType, class iType>
/* implementation defined */ TeamThreadRange(TeamMemberType team, iType count);
template<class TeamMemberType, class iType1, class iType2>
/* implementation defined */ TeamThreadRange(TeamMemberType team, iType1 begin, iType2 end);
```

## Description

 * ```c++
   template<class TeamMemberType, class iType>
   /* Implementation defined */ TeamThreadRange(TeamMemberType team, iType count);
   ```
   Splits the index range `0` to `count-1` over the threads of the team. 
    *  **Arguments**
        * `team`: a handle to the calling team execution context.
        * `count`: index range length. 

    *  **Returns**
        * Implementation defined type.

    *  **Requirements**
        * `TeamMemberType` is a type that models [TeamHandle](TeamHandleConcept)
        * `std::is_integral<iType>::value` is true.
        * Every member thread of `team` must call the operation in the same branch, i.e. it is not legal to have some 
          threads call this function in one branch, and the other threads of `team` call it in another branch.
        * `count >= 0 ` is true;
 
 * ```c++
   template<class TeamMemberType, class iType1, class iType2>
   /* Implementation defined */ TeamThreadRange(TeamMemberType team, iType1 begin, iType2 end);
   ```
   Splits the index range `begin` to `end-1` over the threads of the team. 
    *  **Arguments**
        * `team`: a handle to the calling team execution context.
        * `begin`: index range begin. 
        * `end`: index range end.

    *  **Returns**
        * Implementation defined type.

    * **Requirements**

        * `TeamMemberType` is a type that models [TeamHandle](TeamHandleConcept)
        * `std::is_integral<iType1>::value` is true.
        * `std::is_integral<iType2>::value` is true.
        * Every member thread of `team` must call the operation in the same branch, i.e. it is not legal to have some
          threads call this function in one branch, and the other threads of `team` call it in another branch..
        * `end >= begin ` is true;

  
## Examples

```c++
typedef TeamPolicy<>::member_type team_handle;
parallel_for(TeamPolicy<>(N,AUTO,4), KOKKOS_LAMBDA (const team_handle& team) {
     int n = team.league_rank();
     parallel_for(TeamThreadRange(team,M), [&] (const int& i) {
       A(n,i) = B(n) + i;
     });
     team.team_barrier();
     int team_sum;
     parallel_reduce(TeamThreadRange(team,M), [&] (const int& i, int& lsum) {
       lsum += A(n,i);
     },team_sum);
     single(PerTeam(team),[&] () {
       A_rowsum(n) += team_sum;
     });
});
```

