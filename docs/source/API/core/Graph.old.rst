# What are the semantics of `Kokkos::Graph` ?

What are the allowed semantics of `Kokkos::Graph` ?

Questions:

1. Do we document the allowed semantics for which the user gets covered by `Kokkos` or do we try to enforce the semantics with object states and stuff ?
2. What about the execution space instance ? It seems that `submit` should allow one to be passed.
3. Multi-GPU.
4. runtime aggregate node is still not possible, see https://github.com/kokkos/kokkos/issues/6060.
4. Missing documentation online ?

It should allow functionalities listed in https://www.olcf.ornl.gov/wp-content/uploads/2021/10/013_CUDA_Graphs.pdf, slide 4.

## Usage

How would people use `Kokkos::Graph` ?

### The simplest usage I could come with

The graph is known in advance (at compile time) and can be created in the lambda (*i.e.* not using hidden `impl` stuff).
Once created, the user expects that the graph can be re-submitted several time. The user does not want to add/remove nodes once submitted for the first time (no fancy stuff).
The user does not care about streams whatsoever.

1. Create some `data` in a view, and a `functor` to act on it.
2. Create the `graph` and add a parallel-for `node` using the `functor` acting on `data`.
3. Submit the graph as much as you want.

```c++
template <typename Mem>
struct Functor
{
    Kokkos::View<int*, Mem> data;

    template <std::integer T>
    KOKKOS_FUNCTION
    void operator()(const T index) const { ... <data> ... };
};

int main()
{
    const Kokkos::View<int*, Exec> data(...);

    auto graph = Kokkos::Experimental::create_graph<Exec>([&](auto root) {
        [[maybe_unused]] const auto node = root.then_parallel_for(0, ..., Functor<Mem>{ .data = data });
    });

    graph.submit();
}
```

### More advanced usage

The graph is unknown and cannot be easily/prettily create in the lambda (*e.g.* the user attaches nodes dynamically depending on some complex setup like partitioning).
Once created, the user still expects that the graph can be re-submitted several time.
The user care about streams for orchestration.

We need to use some `impl` stuff for such a case.

```c++
/**
 * Create the graph.
 *
 * 1. Damien said there are other ways to do that w/o using Impl, but I could not find them. It seems that TestGraph.hpp only uses
 *    the Kokkos::Experimental::create_graph that takes a closure.
 *    It seems that 'construct_graph' should somehow be promoted to the public API. Is there any reason not to do so?
 * 2. The execution space instance is not used until the executable graph is launched with 'cudaGraphLaunch'.
 *    Therefore, it's questionnable whether it should be part of the Kokkos::Graph state or not (it's an Impl detail though).
 */
auto graph = Kokkos::Impl::GraphAccess::construct_graph(exec_a);
auto root  = Kokkos::Impl::GraphAccess::create_root_ref(graph);

/**
 * Fill the graph with nodes, according to a complex DAG topology.
 * The nodes might be added conditionally (conditions might change at runtime, e.g. MPI partitioning).
 *
 *       ROOT
 *      /    \
 *     N11    N12
 *     |       | \
 *     N21    N22 N23
 *     \      /   /
 *      \    /   /
 *         N31
 *
 * @todo Add @c if nodes. See also https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes/.
 */
std::vector<generic_node_t> N31_predecessors;

if(condition_branch_1) // branch 1
{
    auto N11 = root.then_parallel_for(...label..., ...policy..., ...body...);
    auto N21 = root.then_parallel_for(...label..., ...policy..., ...body...);
    N31_predecessors.push_back(N21);
}

if(condition_branch_2) // branch 2
{
    auto N12 = root.then_parallel_for(...name..., ...policy..., ...body...);
    auto N22 = root.then_parallel_for(...name..., ...policy..., ...body...);
    auto N23 = root.then_parallel_for(...name..., ...policy..., ...body...);
    N31_predecessors.push_back(N22);
    N31_predecessors.push_back(N23);
}

//! This is currently impossible. See also https://github.com/kokkos/kokkos/issues/6060.
auto N31_ready = Kokkos::Experimental::when_all(N31_predecessors);
auto N31 = N31_ready.then_parallel_for(...name..., ...policy..., ...body...);

/**
 * The topology of the graph has been defined.
 * It now has to be instantiated.
 * According to:
 *  - https://www.olcf.ornl.gov/wp-content/uploads/2021/10/013_CUDA_Graphs.pdf (slide 9)
 *  - https://developer.nvidia.com/blog/employing-cuda-graphs-in-a-dynamic-environment/
 * the topology cannot change once the graph has been instantiated,
 * but the nodes parameters may be updated (cudaGraphExecUpdate).
 */
graph.instantiate(...)

/**
 * Launch the graph on some execution space instance.
 * Re-launch onto another execution space instance. 
 * According to cudaGraphLaunch, a stream is allowed and it makes sense.
 *
 * @todo Check for @c HIP and @c SYCL.
 */
graph.submit(exec_b);
graph.submit(exec_c);
```

## What to do, prioritizing

### Promote `construct_graph` to the public API

This allows for advanced use cases that do not fit well with the current closure-based construction API.

Retrieving the root node should also be promoted to the public API.

### `Kokkos::Graph::instantiate`

**Add** `Kokkos::Graph::instantiate` to the public API.

This allows the user to control when the executable graph gets instantiated.

It can be called only once.

Adding nodes after instantiation is prohibited.

### `Kokkos::Graph::submit`

**Change** the public API to accept an execution space instance.

Note that it is simply used to order the graph launch into some work queue.

### Remove the execution space instance from `Kokkos::Graph` state

The title says it all.

### Allow dynamic aggregate node

**Add** a `Kokkos::Experimental::when_all` that allows for a vector/list of nodes to be passed.

## Go further

We might want to get the design of `Kokkos::Graph` close to `std::execution` (https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html).
