Kokkos::parallel_for(policy_t(exec_A, 0, N), MyAxpby{x, y, alpha, beta});
Kokkos::parallel_for(policy_t(exec_B, 0, N), MyAxpby{z, y, gamma, beta});

exec_B.fence();

Kokkos::parallel_reduce(policy_t(exec_A, 0, N), MyDotp{x, z}, dotp);

exec_A.fence();
