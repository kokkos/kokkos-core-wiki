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
        /// How do we partition ?
        exec_A = start;
        exec_B = Kokkos::partition_space(start, 1);
    }

    auto node_xpy = Kokkos::parallel_for(exec_A, policy(N), MyAxpby{x, y, alpha, beta});
    auto node_zpy = Kokkos::parallel_for(exec_B, policy(N), MyAxpby{z, y, gamma, beta});

    /// In the non-graph case,how do we enforce that e.g. node_zpy is done and launch
    /// the parallel-reduce on the same execution space instance as node_xpy without writing
    /// any additional piece of code ?

    /// No need to fence, because @c Kokkos::when_all will take care of that.
    return Kokkos::parallel_reduce(
        Kokkos::when_all(node_xpy, node_zpy),
        policy(N),
        MyDotp{x, z}, dotp
    );
}

int main()
{
    /// A @c Kokkos execution space instance is a scheduler.
    stdexec::scheduler auto scheduler = Kokkos::DefaultExecutionSpace{};

    /**
     * Start the chain of nodes with an "empty" node, similar to @c std::execution::schedule.
     * Under the hood, it creates the @c Kokkos::Graph.
     * All nodes created from this sender will share a handle to the underlying @c Kokkos::Graph.
     */
    stdexec::sender auto start = Kokkos::Graph::schedule(scheduler);

    /// @c Kokkos::parallel_for would behave much like @c std::execution::bulk.
    stdexec::sender auto my_work = Kokkos::parallel_for(start, policy(N), ForFunctor{...});

    /// Pass our chain to some external library function.
    stdexec::sender auto subgraph = library_stuff(mywork);

    /// Add some work again.
    stdexec::sender auto my_other_work = Kokkos::parallel_scan(subgraph, policy(N), ScanFunctor{...});

    /// @c Kokkos::Graph has a free function for instantiating the underlying graph.
    /// All nodes connected to the same handle (i.e. that are on the same chain) are notified
    /// that they cannot be used as senders anymore,
    /// because they are locked in an instantiated graph. In other words, the chain is a DAG, and it
    /// cannot change anymore.
    stdexec::sender auto executable_chain = Kokkos::Graph::instantiate(my_other_work);

    /// Submission is a no-op if the passed sender is a @c Kokkos execution space instance.
    /// Otherwise, it submits the underlying graph.
    Kokkos::Graph::submit(scheduler, executable_chain)

    ::stdexec::sync_wait(scheduler);

    /// Submit the chain again, using another scheduler.
    /// In essence, what @c Kokkos::Graph::submit can do is pertty much similar to what
    /// @c std::execution::starts_on does. It allows the sender to be executed elsewhere.
    Kokkos::Graph::submit(another_scheduler, executable_chain);
}
