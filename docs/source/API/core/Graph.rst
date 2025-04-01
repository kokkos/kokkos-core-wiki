Graph and related
=================

Usage
-----

:cppkokkos:`Kokkos::Graph` is an abstraction that can be used to define a group of asynchronous workloads that are organised as a direct acyclic graph.
A :cppkokkos:`Kokkos::Graph` is defined separatly from its execution, allowing it to be re-executed multiple times.

:cppkokkos:`Kokkos::Graph` is a powerful way of describing workload dependencies. It is also a good opportunity to present all workloads
at once to the driver, and allow some optimizations [ref].

.. note::

    However, because command-group submission is tied to execution on the queue, without having a prior construction step before starting execution, optimization opportunities are missed from the runtime not being made aware of a defined dependency graph ahead of execution.

For small workloads that need to be sumitted several times, it might save you some overhead [reference to some presentation / paper].

:cppkokkos:`Kokkos::Graph` is specialized for some backends:

* :cppkokkos:`Cuda`: [ref to vendor doc]
* :cppkokkos:`HIP`: [ref to vendor doc]
* :cppkokkos:`SYCL`: [ref to vendor doc] -> https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_graph.asciidoc

For other backends, Kokkos provides a defaulted implementation [ref to file].

Philosophy
----------

As mentioned earlier, the :cppkokkos:`Kokkos::Graph` is first defined, and then executed. In fact, before the graph can be executed,
it needs to be *instantiated*.

During the *instantiation* phase, the topology of the graph is **locked**, and an *executable graph* is created.

In short, we have 3 phases:

1. Graph definition (topology DAG graph)
2. Graph instantiation (executable graph)
3. Graph submission (execute)

"Splitting command construction from execution is a proven solution." (https://www.iwocl.org/wp-content/uploads/iwocl-2023-Ewan-Crawford-4608.pdf)

Use cases
---------

Diamond with closure, don't care about `exec`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a simple diamond-like graph within a closure, not caring too much about execution space instances.

This use case demonstrates how a graph can be created from inside a closure, and how it could look like in the future.
It is a very simple use case.

.. note::

    I'm not sure why we should support the closure anyway. I don't see the benefits of enforcing the
    user to create the whole graph in there.

    See :ref:`no_root_node` for discussion.

.. graphviz::
    :caption: Diamond topology

    digraph diamond {
        A -> B;
        A -> C;
        B -> D;
        C -> D;
    }

.. code-block:: c++
    :caption: Current `Kokkos` pseudo-code.

    auto graph = Kokkos::create_graph([&](auto root){
        auto node_A = root.then_parallel_...(...label..., ...policy..., ...functor...);

        auto node_B = node_A.then_parallel_...(...label..., ...policy..., ...functor...);
        auto node_C = node_A.then_parallel_...(...label..., ...policy..., ...functor...);

        auto node_D = Kokkos::when_all(node_B, node_C).then_parallel_...(...label..., ...policy..., ...functor...);
    });
    graph.instantiate();
    graph.submit()

.. code-block:: c++
    :caption: *à la* P2300 (but really I don't like that because `graph` itself is already a *sender*).

    auto graph = Kokkos::create_graph([&](auto root){
        auto node_A = then(root, parallel_...(...label..., ...policy..., ...functor...));

        auto node_B = then(node_A, parallel_...(...label..., ...policy..., ...functor...));
        auto node_C = then(node_A, parallel_...(...label..., ...policy..., ...functor...));

        auto node_D = then(when_all(node_B, node_C), parallel_...(...label..., ...policy..., ...functor...));
    });
    graph.instantiate();
    graph.submit()

Diamond, caring about `exec`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a simple diamond-like graph, caring about execution space instances. No closure.

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
    :caption: Current `Kokkos` pseudo-code.

    auto graph = Kokkos::create_graph(exec_A, [&](auto root){});
    auto root  = Kokkos::Impl::GraphAccess::create_root_node_ref(graph);

    auto node_A = root.then_parallel_...(...label..., ...policy..., ...functor...);

    auto node_B = node_A.then_parallel_...(...label..., ...policy..., ...functor...);
    auto node_C = node_A.then_parallel_...(...label..., ...policy..., ...functor...);

    auto node_D = Kokkos::when_all(node_B, node_C).then_parallel_...(...label..., ...policy..., ...functor...);

    graph.instantiate();
    exec_A.fence("The graph might make some async to-device copies.");

    graph.submit(exec_B);

.. code-block:: c++
    :caption: *à la* P2300 and defer when `Kokkos` performs internal async to-device copies to the `instantiate` step.

    // Step 1: define graph topology (note that no execution space instance required).
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

.. _no_root_node:

To root or not to root ?
~~~~~~~~~~~~~~~~~~~~~~~~

Currently, the :cppkokkos:`Kokkos::Graph` API would expose to the user a "root node" concept that is not strictly needed
by any backend (but might be needed by the default implementation that works with *sinks*).

I think the "root node" might be confusing. IMO, it should not appear in the API for 2 reasons:

1. It can be misleading, as the user might think it's necessary though I think it's an artifact of how :cppkokkos:`Kokkos::Graph`
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
    :caption: *à la* P2300.

    auto graph = Kokkos::construct_graph();

    auto A1 = Kokkos::then(graph, Kokkos::parallel_...(...));
    auto A2 = Kokkos::then(graph, Kokkos::parallel_...(...));
    auto A3 = Kokkos::then(graph, Kokkos::parallel_...(...));

    auto B = Kokkos::then(Kokkos::when_all(A1, A2, A3), Kokkos::parallel_...(...));

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

This is a step towards https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p2300r10.html#design-sender-adaptor-starts_on.

.. code-block:: c++
    :caption: *à la* P2300.

    // Step 1: construct.
    auto graph = Kokkos::construct_graph();

    auto node_1 = Kokkos::then(graph, ...);
    ...

    // Step 2: instantiate.
    graph.instantiate();

    // Step 3: execute, execute, and again.
    graph.submit(exec_A);
    ...
    graph.submit(exec_C);
    ...
    graph.submit(exec_D);

Interoperability
~~~~~~~~~~~~~~~~

Why interoperability matters (helps adoption of :cppkokkos:`Kokkos::Graph`, extensibility, corner cases):

1. Attract users that already use some backend graph (*e.g.* :code:`cudaGraph_t`) towards `Kokkos`. It helps them transition smoothly.
2. Help user integrate backend-specific graph capabilities that are not part of the :cppkokkos:`Kokkos::Graph` API for whatever reason.

Since `Kokkos` might run some stuff linked to its internals at *instantiation* stage, and since in PR https://github.com/kokkos/kokkos/pull/7240
we decided to ensure that before the submission, the graph needs to be instantiated in `Kokkos`, interoperability implies that the user
relies on `Kokkos` for both *instantiation* and *submission*.

.. graphviz::
    :caption: Dark nodes/edges are added through :cppkokkos:`Kokkos::Graph` API, the rest is pre-existing.

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
    :caption: Interoperability pseudo-code *à la* P2300.

    // The user starts creating its graph with a backend API for some reason.
    cudaGraph_t graph;
    cudaGraphCreate(&graph, ...);

    cudaGraphNode_t A, B1, B2, B3, C3;
    ... create kernel nodes and add dependencies ...

    // But at some point wants interoperability with Kokkos.
    auto kokkos_graph = Kokkos::construct_graph(graph);

    auto C1 = Kokkos::then(Kokkos::when_all(B1, B2), ...);
    auto D1 = Kokkos::then(Kokkos::when_all(C1, C3), ...);

    // The user is now bound to Kokkos for instantiation and submission.
    kokkos_graph.instantiate();
    kokkos_graph.submit();

Interweaving
~~~~~~~~~~~~

When a user does not use :cppkokkos:`Graph`, but calls some external library function that does.

In this case, :code:`submit` really needs to be passed an execution space instance to ensure that the graph
is nicely inserted into the user's kernel queues.

Stated verbosely:

    The stream-based (execution space instance based) approach can co-exist in the same code with
    the graph-based approach, thereby making :cppkokkos:`Graph` a very attractive abstraction.
    A use case in which "at the global level" the code uses a stream-based approach can play well with
    some (possibly external) calls that use :cppkokkos:`Graph` under the hood.

Graph update
~~~~~~~~~~~~

From reading :cppkokkos:`Cuda`, :cppkokkos:`HIP` and :cppkokkos:`SYCL` documentations, all have some *executable graph update* mechanisms.

For instance, disabling a node from host (:code:`hipGraphNodeSetEnabled`) can support complex graphs that might slightly change from one submission to another.

    Updates to a graph will be scheduled after any in-flight executions of the same graph and will not affect previous submissions of the same graph.
    The user is not required to wait on any previous submissions of a graph before updating it.

As the topology is fixed, we can only reasonably update kernel parameters or skip a node.

.. note::

    Todo: in solve take Emil work and say that at compile time we could not reasonnably know what its graph
    would look like. But our own assembly graph could be determined at compile time (knowing the system at stake,
    how we partition it and so on -> still a burden)

.. tikz:: Some iterative loop that needs to seed under some condition, as well as a library call for compute.
   :include: Graph.update.tikz
   :libs: backgrounds, calc, positioning, shapes

Iterative processes
~~~~~~~~~~~~~~~~~~~

Plenty of opportunities for :cppkokkos:`Kokkos::Graph` to lean in:

- iterative solver
- line search in optimization
- you name it

Let's take the `AXPBY` micro-benchmark from https://hihat.opencommons.org/images/1/1a/Kokkos-graphs-presentation.pdf:

.. graphviz::
    :caption: Two `AXPBY` followed by a dot product.

    digraph axpby {
        A[label="axpby"];
        B[label="axpby"];
        C[label="dotp"];
        A->C;
        B->C;
    }

.. literalinclude:: Graph.axpby.kokkos.vanilla.cpp
    :language: c++
    :caption: Vanilla `Kokkos`.

.. literalinclude:: Graph.axpby.kokkos.graph.cpp
    :language: c++
    :caption: Current :cppkokkos:`Kokkos::Graph`.

Runtime graph
~~~~~~~~~~~~~

It can happen that a graph cannot be known at compile time. Examples of programs that could not
determine the control flow completely at compile time:
- MPI partitioning
- BLAS routines and system size
- you name it

Therefore, we must support both pure compile time graphs and runtime graphs.
This implies type-erasure. And this is not possible by default in `std::execution` apparently.

Why/when should I choose :cppkokkos:`Kokkos::Graph`
---------------------------------------------------

Two obvious but different cases:

#. A few kernels, probably small, easily manually stream-managed, submitted several times. Then using :cppkokkos:`Graph`
   will help you reduce kernel launch overheads. TODO: link to A1 A2 A3 B graph.
#. A lot of kernels, very complex DAG, probably not worth it thinking too much how they could be efficiently orchestrated
   if :cppkokkos:`Graph` guarentees that it will take care of that for you.

They also use graphs...
-----------------------

* `PyTorch` https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
* `GROMACS` https://developer.nvidia.com/blog/a-guide-to-cuda-graphs-in-gromacs-2023/

Design choices
--------------

Questions we need to answer before going further in the :cppkokkos:`Graph` refactor.

Dispatching
~~~~~~~~~~~

- Do we allow node policies to have a user-provided execution space instance ?
- When does `Kokkos` makes its to-device dispatching (*e.g.* to global memory) ?

Write a single source code, but allow skipping backend graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We should be able to write a single source code and decide if we want the graph to map to the backend graph or just
execute nodes.

This would greatly benefit adoption, and respect `Kokkos` single source code promise.

Design we would like to agree on
--------------------------------

This should be the kind of design we'd like to have (kind of conforming to P2300).

Might be worth reading: https://docs.nvidia.com/hpc-sdk/archive/23.9/pdf/hpc239c++_par_alg.pdf.

.. literalinclude:: Graph.axpby.kokkos.graph.p2300.cpp
    :language: c++
    :caption: *à la* P2300.

References
----------

* https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf
* https://github.com/intel/llvm/blob/sycl/sycl/doc/syclgraph/SYCLGraphUsageGuide.md
* https://developer.nvidia.com/blog/a-guide-to-cuda-graphs-in-gromacs-2023/
* https://hihat.opencommons.org/images/1/1a/Kokkos-graphs-presentation.pdf


*********************************

Ecrire nos exemples de graphes en P2300 -> compiler + tester

Puis en current kokkos graph -> compiler + tester

Puis en draft el kokkos graph à la p2300 -> à titre de guideline (where we want to go)

* TFE Emil Geleleens example: Solver for matrix L -> backsubstitution for lower triangular
  matrix -> dependencies between unknowns to speed up things -> creates a graph with "as many nodes
  there are unknowns" (with variants, but whatsoever we get many nodes) in the input matrix L
  -> his work is not about efficiently launchign this graph and in fact he did it with manual kernel launches
  -> could be nice to use kokkos graph to focus on other things
  -> there might be some sweet spot above which the backend graph makes sense (cost of instantiate and launch)

=> repo privé "uliegecsm/kokkos-graph-p2300"


One blcoker (https://docs.nvidia.com/cuda/pdf/CUSPARSE_Library.pdf): we would need to use graph capture
to embed the cu solver into our graph...

    Most of the cuSPARSE routines can be optimized by exploiting CUDA Graphs capture and
    Hardware Memory Compression features.
    More in details, a single cuSPARSE call or a sequence of calls can be captured by a CUDA
    Graph and executed in a second moment. This minimizes kernels launch overhead and allows
    the CUDA runtime to optimize the whole workflow. A full example of CUDA graphs capture
    applied to a cuSPARSE routine can be found in cuSPARSE Library Samples - CUDA Graph.



Meeting notes
=============

0. Do you know :cppkokkos:`Kokkos::Graph` ?

   :cppkokkos:`Kokkos::Graph` is an abstraction of a DAG of asynchronous workloads that maps to a backend graph,
   or to the defaulted implementation.

   Advantages of the graph: asynchronous management done by the backend driver + launch overhead reduces especially
   when submitting many times.

   .. figure:: Graph.kokkos.3.paper.jpg

     Example from Kokkos 3 paper.

1. We want to refactor the public API of :cppkokkos:`Kokkos::Graph` so that it feels more like `std::execution` (P2300).

   We could think of a graph (e.g. :cppkokkos:`Kokkos::Graph`) as a **multi-shot sender chain** (?).

    .. code-block:: c++
        :caption: Old way

        child = parent.then_parallel_(policy, body);

    .. code-block:: c++
        :caption: P2300-alike way

        child = parallel_for(parent, policy, body);  // usual
        child = parent | parallel_for(policy, body); // piping

   This seems to be an easy step. A few wrappers could be used in a first step to "transport"
   the P2300-alike way arguments to the old way (thereby keeping the `Kokkos::Graph` implementation
   untouched).

2. Deeper refactoring of :cppkokkos:`Kokkos::Graph`:

   * Should the nodes of the graph be senders ? Or should `Kokkos` nodes and graph
     be wrapped in an adaptor-like API to remain an implementation detail hidden to the user ?
     "P2300 nodes" would then have handlers to their :cppkokkos:`Kokkos::Impl` (nodes and graph) counterparts.
     How is this implemented in `HPX` ? When creating a sender, is there some under-the-hood implementation
     class that maps to some `HPX` pre-existing internals ?
   * Current :cppkokkos:`Kokkos::Graph` restrictions:
      - All nodes are targeting the same backend :math:`\implies` only one scheduler type can be used.
      - The chain cannot contain `transfer`, `starts_on`, and so. The scheduling is left to `Kokkos` through
        :cppkokkos:`Kokkos::Graph::submit(exec)`.


https://accu.org/journals/overload/29/164/teodorescu/