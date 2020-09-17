Kokkos Graphs
=============

This document is intended to introduce and justify the design of the Kokkos Graph interface. First we'll focus on the user interface design, why certain restrictions on the frontend are in place, and how the structure of the frontend API enforces those restrictions.  Then we'll discuss the design of the backend-specific customization points and justify this structure based on the needs of various execution model features that we expect the Kokkos Graph interface to support.  Finally, we'll talk briefly about some of the implementation details in the code that connects the frontend to the various backend customization points.

Motivation
----------

Many of Kokkos's large application users have performance that is pretty much entirely bounded by kernel launch latency on GPUs. A number of execution models have introduced abstractions that allow us to amortize this overhead over several related kernels. The first such execution model abstraction is Cuda Graphs, and we expect (preliminarily, at least) that many other execution models that Kokkos supports with backends will expose this sort of functionality in a similar fashion.  Thus, while careful attention was payed to avoiding overfitting to the Cuda Graphs execution model, exposing pieces of that model through abstractions that we expect to be reasonably performance portable was a primary driver of the Kokkos Graph interface design, both frontend and backend.


Frontend API Design
-------------------

As with all of the Kokkos frontend, the primary goal of the Kokkos Graph frontend is to present a restricted programming model that imposes the intersection of restrictions required for performance portability across execution models while collecting the union of the information those models need to execute the program efficiently. In this case, those restrictions start with the Cuda Graphs API.

### Restrictions and Necessary Information

Cuda Graphs is a lazy API for amortizing the launch latency of a group of (potentially execution-dependency-connected) fork-join parallel kernels.  Once constructed, the graph can be relaunched multiple times with significantly increase efficiency relative to independent kernel launches, assuming (mostly) there is no need for modification of the dependency structure or the kernels themselves.  Thus, a sufficient frontend programming model for this functionality must impose at least these _restrictions_:

- The representation of the graph that can be executed cannot be modified.
- The representation of the graph that can be modified cannot be executed.
- Circular dependencies cannot be introduced between nodes on the graph.
- The representation of kernels or nodes in the graph should not be modifiable once connected to the graph.
- Currently, because of uncertainty surrounding the efficiency of host nodes in Cuda Graphs, it must be clear that any code executed outside of a kernel in the graph will not be repeated when the graph is submitted for execution.

The last of these in particular was a strong motivating factor for presenting an explicit interface that requires users to directly express execution dependencies (rather than an implicit interface that determines such structure through recording, capture, or some other implicit mechanism).

Some of the _required information_ for the Cuda Graph execution model includes:

- The beginning (`cudaGraphCreate()`) and end (`cudaGraphInstantiate()`) of the graph construction phase, during which the graph cannot be executed
- The beginning (`cudaGraphInstantiate()`) and end (`cudaGraphExecDestroy()`, `cudaGraphDestroy()`) of the period during which the graph can be (repeatedly) executed without being modified.
- In theory, this exclusively-modify-then-exclusively-execute phase pattern can be repeated multiple times during the lifetime of a graph (`cudaGraphExecDestroy()` to start a new modify phase, things like `cudaGraphKernelNodeSetParams()` or `cudaGraphRemoveDependencies()` to modify the graph, and then `cudaGraphInstantiate()` again to switch back to the execute phase).  We've chosen not to expose this functionality until the performance advantages (relative to simply reconstructing the graph from scratch when it needs modification) and support for this pattern in other execution models becomes clearer.
- In theory, some limited modifications to the graph can be made during the execute phase (e.g., using `cudaGraphExecKernelNodeSetParams()`). We've omitted these to avoid overfitting to the Cuda Graphs model, at least until the execution models that other backends will use for this sort of functionality become clearer.

### Design decisions

#### Phase separation and lifetimes

C++ interfaces typically express things like bounded phases of an underlying entity using object lifetimes. When these lifetimes need to strictly not overlap, it is often helpful to introduce a scope into the programming interface structure

```c++
auto runner = create_graph([=](auto builder) {
  /* graph construction phase */
});
/* graph execution phase */
```

where the lifetimes of the two phases are tied to objects named `builder` and `runner`, respectively.  The use of a copy capture minimizes the likelihood of the `builder` object escaping the scope representing the construction phase, at least as long as deep constness is used by user-level abstractions (like classes) that may hold a `builder`.  Since the closure is executed immediately and synchronously, assertions about the lifetime of the `builder` can also be made at runtime to ensure the strict closure of the construction phase.

#### Reference or value semantics?

As always, there is the interesting question of whether the graph builder (and corresponding representations of the nodes during the construction phase) and the executable graph objects should have reference semantics or value semantics. It seems that a pretty obvious starting point is consistency: either all such representations should have value semantics, or all such representations should have reference semantics. There are exceptions to this advice, but I don't see any reason to  One place I usually start with this sort of design decision is the copy constructor: what would a value semantic copy constructor have to do, when would it be invoked, and in what sorts of situations would these entities be owned or held (e.g., as members in what sorts of classes)?  Probably the easiest of these abstractions in the interface to reason about is the graph node abstraction. A kernel node with value semantics would need to own the kernel that the node represents. Such an entity would need to copy the instance of the kernel it owns (or be non-copyable, which presents its own issues), which would copy the functor. But that functor likely interacts with data via `View`s, which have reference semantics! This leads to surprising behavior. Furthermore, a graph node provides a representation of a dependency, and in the presence of a task graph that forks, for instance, it makes sense to be able to pass separate references to various parts of the application that construct and manage independent siblings in the task graph. In addition, while the Cuda Graph interface provides a way to copy a graph itself, it provides no direct way to create copies of individual nodes or of the executable representation of the graph without reconstructing those entities.  Given all of this, it seems natural for the first-class entities in the Kokkos Graph interface to have reference semantics. 

#### Graph construction

There are a lot of lazy graph construction models that make it structurally impossible to build cycles. For this interface we went with a familiar `a.then(b)` model, in an attempt to avoid introducing unnecessary complexity. Several other models have some more attractive features, but most lack the widespread familiarity of simple continuation passing models.  Thus, the goal was to present a graph construction interface something like:

```c++
auto runner = create_graph([=](auto builder) {
  auto n1 = builder.parallel_for(/*...*/);
  auto n2 = n1.then_parallel_reduce(/*...*/);
  n2.then_parallel_scan(/*...*/);
});
```

with a `when_all()` member function for joins:

```c++
auto runner = create_graph([=](auto builder) {
  auto n1 = builder.parallel_for(/*...*/);
  auto n2 = builder.parallel_reduce(/*...*/);
  builder.when_all(n1, n2).then_parallel_scan(/*...*/);
});
```

The free function `Kokkos::when_all()` already does something different, though we could reclaim this name at some point since the `builder` itself adds no information that can't be obtained from the nodes.

##### Implicit or explicit future sharing?

In this area, there is often some debate about whether or not future-like entities ("graph nodes" in this model) should be single-shot by default or shared by default. We've opted for the latter, mostly because the fork-join semantics of each node in our model don't afford any additional opportunities for fusion or similar optimizations that could be derived from unique successor information.

#### Graph execution

There's no real reason to complicate this. Graphs are tied to an execution space instance at construction time, so the only additional information that needs to be conveyed (in addition to lifetime) is when (in the sequence represented by the execution space instance) how many times the graph should be executed. Something very simple should suffice:

```c++
auto runner = create_graph(/*...*/);
runner.submit();
// no need to do anything unusual here to ensure completion either:
runner.get_execution_space().fence();
```

#### Type Erasure

The goal of this structure is to enable compile-time kernel fusion when the structure of the application permits it while providing a straightforward interface for situations where 

TODO


Backend Customization Point Design
----------------------------------

As with much of Kokkos, the backend customization point design for Kokkos graphs is centered around public interface types (`Graph`, `GraphNodeRef`) with reference semantics that have shared ownership of private implementation types (`GraphImpl` and `GraphNodeImpl`) that can be customized on a per execution space basis.  Additionally, `GraphNodeImpl` contains a lot of boilerplate common to all (or at least most) backends—mostly relating to the intricacies of type erasure and type preservation for the purposes of things like kernel fusion, so a couple of customizable base class templates are included in the `GraphNodeImpl` inheritance hierarchy: `GraphNodeBackendSpecificDetails` and `GraphNodeDetailsBeforeTypeErasure`. This is not totally dissimilar to the customization point structure where `TeamPolicy` inherits from `TeamPolicyInternal`, for instance, and much of the (existing) Kokkos tasking interface also uses a "common boilerplate inherits from customizable base" idiom. Collecting information about the kernel itself and customizations thereof is done by specializing `GraphNodeKernelImpl`, which typically inherits from the appropriate `Impl::Parallel*` specialization for code reuse purposes (though is not required to; the helper trait `PatternImplSpecializationForTag` is used to do this without code duplication).

### Non-movable, non-copyable pimpl types

As is often the case with the pimpl idiom, the implementation types in the Graph interface spend their entire lifetime in the memory allocated by a shared pointer (specifically, in this case, actual `shared_ptr`), and so they don't need to be movable or copyable. Having a consistent address for the entire lifetime of the implementation object has other advantages also, though, particularly when it comes to graphs.  Implementations can easily refer to non-owned objects like predecessors with simple raw pointers (the `ObservingRawPtr` alias template is used for clarity, though) and customization points can safely point into the object hierarchy they're a part of. None of these objects are publicly accessible, of course, so the user perspective on non-movability and non-copyability of these types is irrelevant.

### By example: Walk-through of `GraphNodeRef::then_parallel_for()`

Starting from the implementation of `GraphNodeRef::then_parallel_for()`, the implementation [applies an execution policy property](https://github.com/dhollman/kokkos/blob/b0188e7de46b30696ae540d35e9598de58baadfd/core/src/Kokkos_GraphNode.hpp#L296-L297) that can be ignored by graph-oblivious layers of the software stack but can be detected by, say, the Cuda kernel launch implementation. Then, the implementation [constructs a specialization](https://github.com/dhollman/kokkos/blob/b0188e7de46b30696ae540d35e9598de58baadfd/core/src/Kokkos_GraphNode.hpp#L300-L306) of `GraphNodeKernelImpl` with the [appropriate tag](https://github.com/dhollman/kokkos/blob/b0188e7de46b30696ae540d35e9598de58baadfd/core/src/Kokkos_GraphNode.hpp#L303) to indicate that the kernel pattern is a `parallel_for` (note that this has nothing to do with the user-defined dispatch tag that is part of the policy or the `KernelInGraphProperty` that is also part of the policy).  The kernel is constructed from the policy, the execution space instance, the user-defined functor, and the kernel name.  The rest of the process is common to all `then_parallel_*()` calls, so the [private method `_then_kernel()` gets called](https://github.com/dhollman/kokkos/blob/b0188e7de46b30696ae540d35e9598de58baadfd/core/src/Kokkos_GraphNode.hpp#L304) with the kernel instance just constructed.

Given a kernel by one of the `then_parallel_*()` method templates, `_then_kernel()` [creates a `GraphNodeImpl` instance](https://github.com/dhollman/kokkos/blob/b0188e7de46b30696ae540d35e9598de58baadfd/core/src/Kokkos_GraphNode.hpp#L186-L191) from the [execution space instance](https://github.com/dhollman/kokkos/blob/b0188e7de46b30696ae540d35e9598de58baadfd/core/src/Kokkos_GraphNode.hpp#L187), the [kernel instance](https://github.com/dhollman/kokkos/blob/b0188e7de46b30696ae540d35e9598de58baadfd/core/src/Kokkos_GraphNode.hpp#L189), and the [predecessor](https://github.com/dhollman/kokkos/blob/b0188e7de46b30696ae540d35e9598de58baadfd/core/src/Kokkos_GraphNode.hpp#L191), which is the current `GraphNodeRef` instance (i.e., that's what `then` means); the constructor uses disambiguator tags for clarity, as is common with non-movable and non-copyable types that couldn't create named constructors through static member functions before C++17.  We use `new` here with the `shared_ptr` constructor rather than `make_shared` because we need to have a custom deleter. As [this comment](https://github.com/dhollman/kokkos/blob/d12ad60f56c0aeb1dfca071719e706fc67ab8354/core/src/impl/Kokkos_GraphNodeImpl.hpp#L80-L86) attempts to explain, we essentially store the function pointers that would compose the vtable of the `GraphNodeImpl` hierarchy in the object representation itself. This is done with the goal of reducing binary bloat, since the number of vtable entries scales with the number of user-defined kernels, and the cost of a function pointer indirection (and storage) here is irrelevant at this granularity.  At one point in the history of this implementation, there were other function pointers for functions with access to the concrete types of the kernel and the predecessor that essentially amounted to a multiple dispatch to enable type un-erasure for things like kernel fusion, but those have been removed for now simplicity.  They may be added back in later once we explore how we want to do such optimizations.

For consistency and encapsulation, the `make_node_shared_ptr_with_deleter()` [static member function](https://github.com/dhollman/kokkos/blob/bf4e2f32316984eb582e32a70e46556b424b0e25/core/src/impl/Kokkos_GraphImpl.hpp#L94-L103) of the [attorney class](https://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/Friendship_and_the_Attorney-Client).  (As an aside: attorney access isn't really necessary in this one case, because [the deleter](https://github.com/dhollman/kokkos/blob/d12ad60f56c0aeb1dfca071719e706fc67ab8354/core/src/impl/Kokkos_GraphNodeImpl.hpp#L107-L110) needs to be public anyway so that the internals of `shared_ptr` can access it, but if we decide to use something other than a shared pointer, it's good to have the implementation in one place anyway and to have that be a friend of `GraphNodeImpl`).

Getting back to `GraphNodeRef::_then_kernel()`, the implementation constructs a `GraphNodeRef` instance to represent the new node to the user. This needs to be done through an attorney so that the constructor can be private (because `GraphNodeRef` is a public-facing class template that the user isn't allowed to construct that way), and the [static member function template `GraphAccess::make_graph_node_ref()`](https://github.com/dhollman/kokkos/blob/bf4e2f32316984eb582e32a70e46556b424b0e25/core/src/impl/Kokkos_GraphImpl.hpp#L105-L117) does this (it's not sufficient to just make `GraphNodeRef` a friend template of itself, because other backend implementation types also need to call this constructor). After [making the `GraphNodeRef` instance](https://github.com/dhollman/kokkos/blob/b0188e7de46b30696ae540d35e9598de58baadfd/core/src/Kokkos_GraphNode.hpp#L183) to return to the user, the customization point design I've chosen dictates that the [implementation calls the `add_node` customization point](https://github.com/dhollman/kokkos/blob/b0188e7de46b30696ae540d35e9598de58baadfd/core/src/Kokkos_GraphNode.hpp#L195) on the `GraphImpl` instance associated with `this` node (i.e., the new node and its predecessor are part of the same graph). The argument to `add_node` is a `shared_ptr<GraphNodeImpl<...>>` referencing the new node just created. After that, the customization point design dictates that the `add_predecessor()` customization point [will be called](https://github.com/dhollman/kokkos/blob/b0188e7de46b30696ae540d35e9598de58baadfd/core/src/Kokkos_GraphNode.hpp#L198) with the current node and its predecessor (i.e., `*this`). Note that in this way, the maximum type information available to the call site is conveyed to the backend, which is free to use as much or as little of that information as it needs. The new `GraphNodeRef` instance is then [returned back to the user](https://github.com/dhollman/kokkos/blob/b0188e7de46b30696ae540d35e9598de58baadfd/core/src/Kokkos_GraphNode.hpp#L200).


### `GraphImpl`

Syntactically, the requirements on a backend's implementation of `GraphImpl` are essentially:

```c++
template <class GI, class Ex, class GNI, class GNR, class... GNRs>
concept _graph_impl = // exposition only
  is_specialization_of_v<GI, GraphImpl> && // yes, is_specialization_of_v is a terrible concept. I know.
  is_specialization_of_v<GNI, GraphNodeImpl> &&
  is_specialization_of_v<GNR, GraphNodeRef> &&
  (is_specialization_of_v<GNRs, GraphNodeRef> && ...) &&
  requires (GI& gi, std::shared_ptr<GNI> gni_ptr, GNR gnr, GNRs... gnrs) {
    requires Kokkos::ExecutionSpace<typename GI::execution_space>
      && std::constructible_from<GI, typename GI::execution_space>;
    gi.submit();
    gi.get_execution_space() -> convertible_to<typename GI::execution_space>;
    gi.add_node(gni_ptr);
    gi.add_predecessor(gni_ptr, gnr);
    { gi.create_root_node_ptr() }
      -> convertible_to<std::shared_ptr<GraphNodeImpl<typename GI::execution_space>>>;
    { gi.create_aggregate_ptr(gnrs...) }
      -> convertible_to<std::shared_ptr<GraphNodeImpl<typename GI::execution_space>>>;
  };
```

#### `GraphImpl<Ex>::add_node()`

Called with a `shared_ptr<GraphNodeImpl<Ex, ...>>` (call it `node`) to indicate that the backend should add `node` to the set of things that need to be executed each time `submit()` is called.  Specifically, "executed" this means that the backend should call the functor of the kernel associated with `node` (via `node.get_kernel()`, which is a convenience function for obtaining the same object at the same address as the `Kernel` passed to the constructor of `GraphNodeBackendDetailsBeforeTypeErasure` during the construction of `node`) according to the policy associated with the kernel (both the policy and functor associations with the kernel are made at construction time of the `GraphNodeKernelImpl` specialization).

#### `GraphImpl<Ex>::add_predecessor()`

Called with a `shared_ptr<GraphNodeImpl<Ex, ...>` (call it `node`) and a `GraphNodeRef<Ex, ...>` (call it `predecessor`) to indicate that the join of the kernel associated with `predecessor` must finish before the launch of the kernel associate with `node` when the graph is submitted. Both `node` and the `GraphNodeImpl<Ex, ...>` associated with `predecessor` must be objects that have already been registered with `add_node()` on the same graph.

#### `GraphImpl<Ex>::create_root_node_ptr()

Called once in the lifetime of the graph, at the beginning of the construction phase, to allow the backend to create a node that is a predecessor (at least transitively) to all nodes in the graph. For consistency, `add_node()` will also be called with the value returned by this customization point.

#### `GraphImpl<Ex>::create_aggregate_ptr()

Called with a set of predecessors of types `GraphNodeRef<Ex, /* multiple types */>...` (call them `predecessors...`). The returned node should have a trivial kernel, but the implementation should still ensure that the join of the kernel in each of the `predecessors...` completes before any successors of the returned node. For consistency, `add_node` will be called with the returned node, and `add_predecessor` will be called with the returned node and each of the arguments to this customization point.

### `GraphNodeKernelImpl`

The implementation only really interacts with most of the rest of the customization points exposed to backends via constructors.  The syntactic requirements on `GraphNodeKernelImpl` specializations are:

```c++
template <class GNK, class Ex, class F, class Pol, class... Args>
concept _graph_node_kernel = // exposition only
  is_specialization_of_v<GNK, GraphNodeKernelImpl> &&
  Kokkos::ExecutionSpace<Ex> &&
  Kokkos::DataParallelFunctor<F> &&
  Kokkos::ExecutionPolicy<Pol> &&
  requires {
    requires Pol::is_graph_kernel::value;
  } &&
  constructible_from<GNK, std::string, Ex, F, Pol, Args...>;
```

### `GraphNodeBackendSpecificDetails`

Currently, the syntactic requirements on `GraphNodeBackendSpecificDetails` customizations are very minimal:

```c++
template <class GNBSI>
concept _graph_node_backend_specific_details = // exposition only
  is_specialization_of_v<GNBSI, GraphNodeBackendSpecificDetails> &&
  default_constructible<GNBSI> &&
  constructible_from<GNBSI, _graph_node_is_root_ctor_tag>;
```

The backend can use this customization point to store whatever data needs to be associated with nodes in the graph that get passed to `add_node()` and `add_predecessor()`.

### `GraphNodeBackendDetailsBeforeTypeErasure`

The syntactic requirements on `GraphNodeBackendDetailsBeforeTypeErasure` customizations are also pretty minimal. Just like `GraphNodeBackendSpecificDetails`, the implementation basically only imposes requirements on the constructors so that these class templates can be integrated into the `GraphNodeImpl` hierarchy.

```c++
template <class GNBD, class GNBSI, class GNK, class GNR, class Ex>
concept _graph_node_backend_details_before_type_erasure = // exposition only
  is_specialization_of_v<GNBD, GraphNodeBackendDetailsBeforeTypeErasure> &&
  _graph_node_backend_specific_details<GNBSI> &&
  _graph_node_kernel<GNK> &&
  is_specialization_of_v<GNR, GraphNodeRef> &&
  constructible_from<GNBD, Ex, GNK&, GNR, GNBSI&> &&
  constructible_from<GNBD, Ex, _graph_node_is_root_ctor_tag, GNBSI&>;
```




Implementation Notes
--------------------

### `GraphNodeImpl`

As previously alluded to, the structure of `GraphNodeImpl` is complicated a bit by the desire to minimize type erasure when the user can use return type deduction while maintaining a simple and intuitive interface for when the user needs to write the type explicitly.  The basic structure involves three layers with progressively increased type erasure along side the two previously mentioned backend customization points, `GraphNodeBackendSpecificDetails` and `GraphNodeDetailsBeforeTypeErasure`.  The most-derived version has the concrete type of the kernel _and_ the concrete type of the predecessor. The next layer up erases the predecessor, and the top layer (which the user gets a reference to when they write `GraphNodeRef<ExecSpace>`) erases both the kernel and the predecessor information. 

TODO finish this

### `GraphNodeKernelImpl`

### Host Backends


### Cuda Backend


