auto graph = Kokkos::construct_graph();

auto node_xpy = Kokkos::then(graph, Kokkos::parallel_for(N, MyAxpby{x, y, alpha, beta}));
auto node_zpy = Kokkos::then(graph, Kokkos::parallel_for(N, MyAxpby{z, y, gamma, beta}));

auto node_dotp = Kokkos::then(
    Kokkos::when_all(node_xpy, node_zpy),
    Kokkos::parallel_reduce(N, MyDotp{x, z}, dotp)
);

graph.instantiate();

graph.submit(exec_A);

exec_A.fence();
