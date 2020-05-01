## Top Level Execution Policies

|Policy  |Description                  |
|:---------|:----------------------------|
|[RangePolicy](Kokkos%3A%3ARangePolicy) | Each iterate is an integer in a contiguous range |
|[MDRangePolicy](Kokkos%3A%3AMDRangePolicy) | Each iterate for each rank is an integer in a contiguous range |
|[TeamPolicy](Kokkos%3A%3ATeamPolicy) | Assigns to each iterate in a contiguous range a team of threads |

## Nested Execution Policies
|Policy  |Description                  |
|:---------|:----------------------------|
|[TeamThreadRange](Kokkos%3A%3ATeamThreadRange) | Used inside of a TeamPolicy kernel to perform nested parallel loops |
