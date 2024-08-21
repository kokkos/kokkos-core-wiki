Graph and related
=================

Usage
-----

:code:`Kokkos::Graph` is an abstraction that can be used to define a group of asynchronous workloads that are organised as a direct acyclic graph.
A :code:`Kokkos::Graph` is defined separatly from its execution, allowing it to be re-executed multiple times.

:code:`Kokkos::Graph` is a powerful way of describing workload dependencies. It is also a good opportunity to present all workloads
at once to the driver, and allow some optimizations [ref].

.. note::

    However, because command-group submission is tied to execution on the queue, without having a prior construction step before starting execution, optimization opportunities are missed from the runtime not being made aware of a defined dependency graph ahead of execution.

For small workloads that need to be sumitted several times, it might save you some overhead [reference to some presentation / paper].

:code:`Kokkos::Graph` is specialized for some backends:

* :code:`Cuda`: [ref to vendor doc]
* :code:`HIP`: [ref to vendor doc]
* :code:`SYCL`: [ref to vendor doc] -> https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_graph.asciidoc

For other backends, Kokkos provides a defaulted implementation [ref to file].

Philosophy
----------

As mentioned earlier, the :code:`Kokkos::Graph` is first defined, and then executed. In fact, before the graph can be executed,
it needs to be *instantiated*.

During the *instantiation* phase, the topology of the graph is **locked**, and an *executable graph* is created.

In short, we have 3 phases:

1. Graph definition (topology DAG graph)
2. Graph instantiation (executable graph)
3. Graph submission (execute)

"Splitting command construction from execution is a proven solution." (https://www.iwocl.org/wp-content/uploads/iwocl-2023-Ewan-Crawford-4608.pdf)

Basic example
-------------

This example showcases how three workloads can be organised as a :code:`Kokkos::Graph`.

Workloads A and B are independent, but workload C needs the completion of A and B.

.. code-block:: cpp

    int main()
    {
        auto graph = Kokkos::Experimental::create_graph<Exec>([&](auto root) {
            const auto node_A = root.then_parallel_for(...label..., ...policy..., ...body...);
            const auto node_B = root.then_parallel_for(...label..., ...policy..., ...body...);
            const auto ready  = Kokkos::Experimental::when_all(node_A, node_B);
            const auto node_C = ready.then_parallel_for(...label..., ...policy..., ...body...);
        });

        for(int irep = 0; irep < nrep; ++irep)
            graph.submit();
    }

Advanced example
----------------

To be done soon.

References
----------

* https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf
* https://github.com/intel/llvm/blob/sycl/sycl/doc/syclgraph/SYCLGraphUsageGuide.md
* https://developer.nvidia.com/blog/a-guide-to-cuda-graphs-in-gromacs-2023/


Use cases
---------

Diamond with closure, don't care about `exec`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a simple diamond-like graph within a closure, no caring about execution space instances.

This use case demonstrates how a graph can be created from inside a closure, and how it could look like in the future.
It is a very simple use case.

Note that I'm not sure why we should support the closure anyway.

.. graphviz::
    :caption: Diamond topology

    digraph diamond {
        A -> B;
        A -> C;
        B -> D;
        C -> D;
    }

.. code-block:: c++
    :caption: Current pseudo-code

    auto graph = Kokkos::create_graph([&](const auto& root){
        auto node_A = root.then_parallel_...(...label..., ...policy..., ...functor...);

        auto node_B = node_A.then_parallel_...(...label..., ...policy..., ...functor...);
        auto node_C = node_A.then_parallel_...(...label..., ...policy..., ...functor...);

        auto node_D = Kokkos::when_all(node_B, node_C).then_parallel_...(...label..., ...policy..., ...functor...);
    });
    graph.instantiate();
    graph.submit()

.. code-block:: c++
    :caption: P2300 (but really I don't like that because `graph` itself is already a *sender*)

    auto graph = Kokkos::create_graph([&](const auto& root){
        auto node_A = then(root, parallel_...(...label..., ...policy..., ...functor...));

        auto node_B = then(node_A, parallel_...(...label..., ...policy..., ...functor...));
        auto node_C = then(node_A, parallel_...(...label..., ...policy..., ...functor...));

        auto node_D = then(when_all(node_B, node_C), parallel_...(...label..., ...policy..., ...functor...));
    });
    graph.instantiate();
    graph.submit()

Diamond, caring about `exec`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a simple diamond-like graph, caring about execution space instances.

This use case demonstrates how a graph can be created without a closure, and how it could look like in the future.
It also focuses on where steps occur.

Graph topology is known at compile, thus enabling a lot of optimizations (kernel fusion might be one).

.. graphviz::
    :caption: Diamond topology

    digraph diamond {
        A -> B;
        A -> C;
        B -> D;
        C -> D;
    }

.. code-block:: c++
    :caption: Current pseudo-code

    auto graph = Kokkos::create_graph(exec_A, [&](const auto& root){});
    auto root  = Kokkos::Impl::GraphAccess::create_root_node_ref(graph);

    auto node_A = root.then_parallel_...(...label..., ...policy..., ...functor...);

    auto node_B = node_A.then_parallel_...(...label..., ...policy..., ...functor...);
    auto node_C = node_A.then_parallel_...(...label..., ...policy..., ...functor...);

    auto node_D = Kokkos::when_all(node_B, node_C).then_parallel_...(...label..., ...policy..., ...functor...);

    graph.instantiate();
    exec_A.fence("The graph might make some async to-device copies.");
    graph.submit(exec_B);

.. code-block:: c++
    :caption: P2300 + defer when Kokkos performs internal async to-device copies

    // Step 1: define topology (no execution space instance required)
    auto graph = Kokkos::create_graph<execution_space>();

    auto node_A = then(graph, parallel_...(...label..., ...policy..., ...functor...));

    auto node_B = then(node_A, parallel_...(...label..., ...policy..., ...functor...));
    auto node_C = then(node_A, parallel_...(...label..., ...policy..., ...functor...));

    auto node_D = then(when_all(node_B, node_C), parallel_...(...label..., ...policy..., ...functor...));

    // Step 2: instantiate (execution space instance required by both backend and Kokkos internals)
    graph.instantiate(exec_A);
    exec_A.fence();

    // Step 3: execute
    graph.submit(exec_B)

No "root" node
~~~~~~~~~~~~~~

Currently, the :code:`Kokkos::Graph` would expose to the user a "root node" concept that is not needed
by any backend (but might be needed by the default implementation that works with *sinks*).

The "root node" might be confusing. It sould not appear in the API for 2 reasons:

1. It can be misleading, as the user might think it's necessary though I think it's an artifact of how :code:`Kokkos::Graph`
   is currently implemented for graph construction, and because of the *sink*-based defaulted implementation.
2. With P2300, it's clear that *root* is an empty useless sender that can be thrown away at compile time.

.. graphviz::
    :caption: No root node.

    digraph no_root {
        A1 -> B;
        A2 -> B;
        A3 -> B;
    }

.. code-block:: c++
    :caption: P2300

    auto graph = construct_graph();

    auto A1 = then(graph, ...);
    auto A2 = then(graph, ...);
    auto A3 = then(graph, ...);

    auto B = then(when_all(A1, A2, A3), ...);

Complex DAG topology
~~~~~~~~~~~~~~~~~~~~

Any complex-but-valid DAG topology should work.

.. graphviz::
    :caption: A complex DAG

    digraph complex_dag {

        A1 -> B1;
        A1 -> B2;
        A1 -> B3;
        A2 -> B1;
        A2 -> B3;
        A3 -> B4;
        
        B1 -> C1;
        B3 -> C1;
        
        B2 -> C2;
        B4 -> C2;
        
        // Enfore ordering of nodes with invisible edges.
        {
            rank = same;
            edge[ style=invis];
            B1 -> B2 -> B3 -> B4 ;
            rankdir = LR;
        }
    }

Changing scheduler
~~~~~~~~~~~~~~~~~~

This is the purpose of PR https://github.com/kokkos/kokkos/pull/7249, and should be further documented.

Towards https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html#design-sender-adaptor-starts_on.

.. code-block:: c++

    auto graph = construct()

    auto node_1 = ...

    ...

    graph.instantiate();

    graph.submit(exec_A);

    ...

    graph.submit(exec_C);

    ...

    graph.submit(exec_D);

Interoperability
~~~~~~~~~~~~~~~~

Why interoperability matters (helps adoption of :code:`Kokkos::Graph`, extensibility, corner cases):

1. Attract users that already use some backend graph (*e.g.* `cudaGraph_t`) towards `Kokkos`. It helps them transition smoothly.
2. Help user integrate backend-specific graph capabilities that are not part of the :code:`Kokkos::Graph` API for whatever reason.

Since `Kokkos` might run some stuff linked to its internals at *instantiation* stage, and since in PR https://github.com/kokkos/kokkos/pull/7240
we decided to ensure that before the submission, the graph needs to be instantiated in `Kokkos`, interoperability implies that the user
passes through `Kokkos` for both *instantiation* and *submission*.

.. graphviz::
    :caption: Dark nodes/edges are added through :code:`Kokkos::Graph`.

    digraph interoperability {

        A[color=darksalmon];
        
        B1[color=darksalmon];
        B2[color=darksalmon];
        B3[color=darksalmon];
        
        C3[color=darksalmon];

        A -> B1[color=darksalmon];
        A -> B2[color=darksalmon];
        A -> B3[color=darksalmon];
        
        B3 -> C3[color=darksalmon];
        
        // Enfore ordering of nodes with invisible edges.
        {
            rank = same;
            edge[style=invis];
            B1 -> B2 -> B3 ;
            rankdir = LR;
        }
        
        B1 -> C1;
        B2 -> C1;
        
        C1 -> D1;
        C3 -> D1;
    } 

.. code-block:: c++
    :caption: interoperability pseudo-code P2300

    cudaGraph_t graph;
    cudaGraphCreate(&graph, ...);

    cudaGraphNode_t A, B1, B2, B3, C3;
    ... create kernel nodes and add dependencies ...

    auto kokkos_graph = construct(graph);

    auto C1 = then(when_all(B1, B2), ...);
    auto D1 = then(when_all(C1, C3), ...);

    kokkos_graph.instantiate();
    kokkos_graph.submit();

Graph update
~~~~~~~~~~~~

From reading `Cuda`, `HIP` and `SYCL` documentations, all have some *executable graph update* mechanisms.

For instance, disabling a node from host (:code:`hipGraphNodeSetEnabled`, not in `HIP` yet) can support complex graphs that might slightly change from one submission to another.

    Updates to a graph will be scheduled after any in-flight executions of the same graph and will not affect previous submissions of the same graph.
    The user is not required to wait on any previous submissions of a graph before updating it.

As the topology is fixed, we can only reasonably update kernel parameters.
