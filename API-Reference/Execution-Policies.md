## Top Level Execution Policies

|Policy  |Description                  |
|:---------|:----------------------------|
|[RangePolicy](Kokkos%3A%3ARangePolicy) | Each iterate is an integer in a contiguous range |
|[MDRangePolicy](Kokkos%3A%3AMDRangePolicy) | Each iterate for each rank is an integer in a contiguous range |
|[TeamPolicy](Kokkos%3A%3ATeamPolicy) | Assigns to each iterate in a contiguous range a team of threads |

## Nested Execution Policies

Nested Execution Policies are used to dispatch parallel work inside of an already executing parallel region either dispatched with a [TeamPolicy](Kokkos%3A%3ATeamPolicy) or a
task policy. 

|Policy  |Description                  |
|:---------|:----------------------------|
|[TeamThreadRange](Kokkos%3A%3ATeamThreadRange) | Used inside of a TeamPolicy kernel to perform nested parallel loops split over threads of a team. |
|[TeamVectorRange](Kokkos%3A%3ATeamThreadRange) | Used inside of a TeamPolicy kernel to perform nested parallel loops split over threads of a team and their vector lanes. |
|[ThreadVectorRange](Kokkos%3A%3ATeamThreadRange) | Used inside of a TeamPolicy kernel to perform nested parallel loops with vector lanes of a thread. |
