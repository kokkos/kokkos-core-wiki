16. Graphs
==========

Usage
-----

:cppkokkos:`Kokkos::Graph` is an abstraction that describes
asynchronous workloads organised as a direct acyclic graph (DAG).

Once defined, the graph can be executed many times.

:cppkokkos:`Kokkos::Graph` is specialized for some backends:

* :cppkokkos:`Cuda`
* :cppkokkos:`HIP`
* :cppkokkos:`SYCL`

On these backends, the :cppkokkos:`Kokkos::Graph` specialisations map to the native graph API, namely, the CUDA Graph API, the HIP Graph API, and the SYCL (command) Graph API, respectively.

For other backends, :cppkokkos:`Kokkos::Graph` provides a defaulted implementation.

Execution space instance versus graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Workloads submitted on :cppkokkos:`Kokkos` execution space instances execute *eagerly*, *i.e.*,
once the :cppkokkos:`Kokkos::parallel_` function is called, the workload is immediately launched on the device.

By contrast, the :cppkokkos:`Kokkos::Graph` abstraction follows *lazy* execution,
*i.e*, workloads added to a :cppkokkos:`Kokkos::Graph` are **not** executed *until*
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

Capture
~~~~~~~

It might be necessary to add nodes whose workloads are not
defined using any :cppkokkos:`Kokkos` code to a :cppkokkos:`Kokkos::Graph`.
This is the case *e.g.* when calling external math libraries like `cuBLAS`.

Such a scenario can be encountered in many situations like building and training a neural network,
running a conjugate gradient method, and so on.

Capturing into a :cppkokkos:`Kokkos::Graph` boils down to writing the following snippet:

.. code:: c++

    auto my_captured_node = predecessor.cuda_capture(
        exec,
        [data = data](const Kokkos::Cuda& exec_) { ... }
    );

When the node is added to the :cppkokkos:`Kokkos::Graph`, the workloads are not directly dispatched to the device.
Rather, the backend operations are "saved" for later "reuse" in the *capture* node.

This is achieved by setting the underlying "stream" to "*capture* mode" before invoking the function object,
such that backend operations are saved to the *capture* node instead of being enqueued in the passed execution space instance.
The underlying "stream" is then restored to "no-*capture* mode".

Some important aspects of *capture* are worth pointing out:

1. The function object will be stored by the :cppkokkos:`Kokkos::Graph` instance,
   thereby ensuring that any data bound to the function object is guaranteed to
   stay alive until the graph is destroyed.
2. The execution space instance `exec` associates the captured workloads to a device.
3. The *capture* mechanism will temporarily alter the `exec`'s state by setting the underlying "stream" to "*capture* mode".
   While in "*capture* mode", backend-specific restrictions may apply (see `the Cuda programming guide <https://docs.nvidia.com/cuda/cuda-c-programming-guide/#prohibited-and-unhandled-operations>`_
   for instance).

.. warning::

    When a "stream" is used by many threads, capturing on one thread may affect other threads
    (see `CUDA`'s :cpp:`cudaThreadExchangeStreamCaptureMode` for instance).

For now, *capture* is only supported for the following backends:

.. list-table::

  * - Backend
    - Resources
  * - :cppkokkos:`Cuda`
    - `CUDA Graphs <https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-graphs>`_
  * - :cppkokkos:`HIP`
    - `HIP stream capture <https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/graph_management.html#_CPPv421hipStreamBeginCapture11hipStream_t20hipStreamCaptureMode>`_
  * - :cppkokkos:`SYCL`
    - `SYCL Graph <https://github.com/intel/llvm/blob/ee5e1ca95c78576c1b6f12b1c2d461ef4b537a9b/sycl/doc/extensions/experimental/sycl_ext_oneapi_graph.asciidoc?plain=1#L167-LL170>`_

.. note::

    The :cppkokkos:`SYCL` documentation will use *recording* instead of *capture*, but it is essentially the same thing.

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

The following snippet defines, instantiates and submits a :cppkokkos:`Kokkos::Graph`
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

Capture of a `cuBLAS` call
~~~~~~~~~~~~~~~~~~~~~~~~~~

This example shows how to create a node that captures a `cuBLAS` call.
It also demonstrates how data is kept alive during
the whole lifetime of the :cppkokkos:`Kokkos::Graph` (*e.g.* the `cuBLAS` handle).

.. literalinclude:: examples/graph_capture.cpp
   :language: c++
