Kokkos has support for lightweight task-based programming, which is currently pretty limited but we plan to substantially expand in the future.

Will Kokkos Tasking work for my problem?
========================================

Not all task-based problems are a good fit for the current Kokkos approach to tasking.  Currently, the tasking interface in Kokkos is  targetted at problems with kernels far too small to overcome the inherent overhead of top-level Kokkos data parallel launches—that is, small but plentiful data parallel tasks with a non-trivial dependency structure.  For tasks that fit this general scale model but have (very) trivial dependency structures, it may be easier to use [hierarchical parallelism](HierarchicalParallelism), potentially with a `Kokkos::Schedule<Dynamic>` scheduling policy (see, for instance, [this page](Kokkos%3A%3ARangePolicy)) for load balancing if necessary. 

Basic Usage
===========

Fundamentally, task parallelism is just another form of parallelism in Kokkos.  The same general idiom applies as for ordinary [parallel dispatch](ParallelDispatch):

![parallel-dispatch](https://github.com/kokkos/ProgrammingGuide/blob/master/figures/parallel-dispatch.png)

Similarly, for tasking, we have:

![task-dispatch](https://github.com/kokkos/ProgrammingGuide/blob/master/figures/task-dispatch.png)
