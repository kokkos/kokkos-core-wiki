auto graph = Kokkos::Experimental::create_graph(exec_A, [&](auto root){
    auto node_xpy = root.then_parallel_for(N, MyAxpby{x, y, alpha, beta});
    auto node_zpy = root.then_parallel_for(N, MyAxpby{z, y, gamma, beta});

    auto node_dotp = Kokkos::Experimental::when_all(node_xpy, node_zpy).then_parallel_reduce(
        N, MyDotp{x, z}, dotp
    )
});

graph.submit(exec_A);

exec_A.fence();
