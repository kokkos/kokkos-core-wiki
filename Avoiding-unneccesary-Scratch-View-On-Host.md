In some situations where it can be useful to leverage Kokkos' Level 0 Scratch Space with a TeamPolicy for GPUs, it actually costs performance on the CPU.
While on the GPU the scratch space provides access to cache which otherwise would go unused, on CPUs scratch space lives just in the same memory as everything else, and will be cached like any other data access. While it might still help with avoiding repeated gather or scatter operations, it may just incur extra cost if the scratch is only used to manually cache an already contiguous chunk of memory. For those cases simply taking a subview on the host may be sufficient.

Consider a matrix vector multiply:

```c++
using policy_t = Kokkos::TeamPolicy<>;
using team_t = typename policy_t::member_type;
using scratch_t = Kokkos::View<double*, typename Kokkos::DefaultExecutionSpace::scratch_space>;
int rows_per_team = 128;

Kokkos::parallel_for(policy_t(N, Kokkos::AUTO), 
  KOKKOS_LAMBDA(const team_t& team_handle) {
  int start_row = team.league_rank() * rows_per_team;
  int end_row = start_row + rows_per_team;
  if(end_row > num_rows) end_row = num_rows;
  for(int row = start_row; row<end_row; row++) {
    Kokkos::
  
});
```