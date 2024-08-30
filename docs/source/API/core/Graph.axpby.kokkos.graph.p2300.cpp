/**
 * This is some external library function to which we pass a sender.
 * The sender might either be a regular @c Kokkos execution space instance
 * or a graph-node-sender-like stuff.
 * The asynchronicity within the function will either be provided by the graph
 * or must be dealt with in the regular way (creating many space instances).
 */
sender library_stuff(sender start)
{
    sender auto exec_A, exec_B;
    
    if constexpr (Kokkos::is_a_sender<sender>) {
        exec_A = exec_B = start;
    } else {
        std::tie(exec_A, exec_B) = Kokkos::partition_space(start, 1, 1);
    }

    auto node_xpy = Kokkos::parallel_for(exec_A, policy(N), MyAxpby{x, y, alpha, beta});
    auto node_zpy = Kokkos::parallel_for(exec_B, policy(N), MyAxpby{z, y, gamma, beta});

    /// No need to fence, because @c Kokkos::when_all will take care of that.
    return Kokkos::parallel_reduce(
        Kokkos::when_all(node_xpy, node_zpy),
        policy(N),
        MyDotp{x, z}, dotp
    );
}

int main()
{
    scheduler auto exec = Kokkos::DefaultExecutionSpace{};

    /**
    * Start the chain of nodes with an "empty" node, similar to @c std::execution::schedule.
    * Under the hood, it creates the @c Kokkos::Graph.
    * All nodes created from this sender will share a handle to the underlying @c Kokkos::Graph.
    */
    sender auto start = Kokkos::construct_empty_node(exec);

    sender auto seeding = Kokkos::parallel_for(start, policy(N), SomeWork{...});

    /// Pass our chain to some external library function.
    sender auto subgraph = library_stuff(seeding);

    sender auto last_action = Kokkos::parallel_scan(subgraph, policy(N), ScanFunctor{...});

    /// @c Kokkos has a free function for instantiating the underlying graph.
    /// All nodes connected to the same handle are notified that they cannot be used as senders anymore,
    /// because they are locked in an instantiated graph.
    sender auto executable_whatever = Kokkos::Graph::instantiate(last_action);

    /// Submission is a no-op if the received sender is an execution space instance.
    /// Otherwise, it submits the underlying graph.
    Kokkos::Graph::submit(my_exec, executable_whatever)

    my_exec.fence();
}
