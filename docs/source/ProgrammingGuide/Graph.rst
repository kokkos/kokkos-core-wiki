Graphs
======

Usage
-----

:cpp:`Kokkos::Graph` is an abstraction that describes
asynchronous workloads organised as a direct acyclic graph (DAG).

Once defined, the graph can be executed many times.

:cpp:`Kokkos::Graph` is specialized for some backends:

* :cpp:`Cuda`
* :cpp:`HIP`
* :cpp:`SYCL`

On these backends, the :cpp:`Kokkos::Graph` specialisations map to the native graph API, namely, the CUDA Graph API, the HIP Graph API, and the SYCL (command) Graph API, respectively.

For other backends, :cpp:`Kokkos::Graph` provides a defaulted implementation.

Execution space instance versus graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Workloads submitted on :cpp:`Kokkos` execution space instances execute *eagerly*, *i.e.*,
once the :cpp:`Kokkos::parallel_` function is called, the workload is immediately launched on the device.

By contrast, the :cpp:`Kokkos::Graph` abstraction follows *lazy* execution,
*i.e*, workloads added to a :cpp:`Kokkos::Graph` are **not** executed *until*
the whole graph is ready and submitted.

Always in 3 phases
~~~~~~~~~~~~~~~~~~

Typically, 3 phases are needed:

1. definition
2. instantiation
3. submission

The *definition* phase consists in describing the workloads: what they do, as well as their dependencies.
In other words, this phase creates a *topological* graph of workloads.

The *instantiation* phase **locks** the topology, *i.e.*, it cannot be changed anymore.
During this phase, the graph will be checked for flaws.
The backend creates an *executable* graph.

The last phase is *submission*. It will execute the workloads, observing their dependencies.
This phase can be run multiple times.

Advantages
~~~~~~~~~~

There are many advantages. Here are a few:

* Since the workloads are described ahead of execution,
  the backend driver and/or compiler can leverage optimization opportunities.
* Launch overhead is reduced, benefitting DAGs consisting of small workloads.

Examples
--------

Diamond DAG
~~~~~~~~~~~

Consider a diamond-like DAG.

.. graphviz::

    digraph diamond {
        A -> B;
        A -> C;
        B -> D;
        C -> D;
    }

The following snippet defines, instantiates and submits a :cpp:`Kokkos::Graph`
for this DAG.

.. code-block:: c++

    auto graph = Kokkos::create_graph([&](auto root) {
        auto node_A = root.then_parallel_for("workload A", ...policy..., ...functor...);

        auto node_B = node_A.then_parallel_for("workload B", ...policy..., ...functor...);
        auto node_C = node_A.then_parallel_for("workload C", ...policy..., ...functor...);

        auto node_D = Kokkos::when_all(node_B, node_C).then_parallel_for("workload D", ...policy..., ...functor...);
    });

    graph.instantiate();

    graph.submit();
