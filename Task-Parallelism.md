Kokkos has support for lightweight task-based programming, which is currently pretty limited but we plan to substantially expand in the future.

Will Kokkos Tasking work for my problem?
========================================

Not all task-based problems are a good fit for the current Kokkos approach to tasking.  Currently, the tasking interface in Kokkos is  targetted at problems with kernels far too small to overcome the inherent overhead of top-level Kokkos data parallel launches—that is, small but plentiful data parallel tasks with a non-trivial dependency structure.  For tasks that fit this general scale model but have (very) trivial dependency structures, it may be easier to use [hierarchical parallelism](HierarchicalParallelism), potentially with a `Kokkos::Schedule<Dynamic>` scheduling policy (see, for instance, [this page](Kokkos%3A%3ARangePolicy)) for load balancing if necessary. 

Basic Usage
===========

Fundamentally, task parallelism is just another form of parallelism in Kokkos.  The same general idiom of pattern, policy, and functor applies as for ordinary [parallel dispatch](ParallelDispatch):

![parallel-dispatch](https://github.com/kokkos/ProgrammingGuide/blob/master/figures/parallel-dispatch.png)

Similarly, for tasking, we have:

![task-dispatch](https://github.com/kokkos/ProgrammingGuide/blob/master/figures/task-dispatch.png)


Task Functor
------------

In most ways, the functor portion of the task parallelism idiom in Kokkos is similar to that for data parallelism.  Task functors haeve the additional requirement that the task output type needs to be provided by the user by giving a nested type (a.k.a. "typedef") named `value_type`:

```c++
struct MyTask {
  using value_type = double;
  template <class TeamMember>
  KOKKOS_INLINE_FUNCTION
  void operator()(TeamMember& member, double& result);
};
```

Similar to [team parallelism](HierarchicalParallelism), the first parameter is the team handle, which has all of the same functionality as the one produced by a `Kokkos::TeamPolicy`, with a few extras.  Like with `Kokkos::parallel_reduce()`, the output is expressed through the second parameter.  Note that there is currently no lambda interface to Kokkos Tasking.

Task Patterns
-------------

The primary analogs of `Kokkos::parallel_for()` and friends for tasking are `Kokkos::task_spawn()` and `Kokkos::host_spawn()`.  They both return a `Kokkos::Future` associated with the appropriate `Scheduler`, but `host_spawn` can only be called from *outside* of a task functor, while `task_spawn` can only be called from *inside* of one.

Task Policies
-------------

There are currently two task policies in Kokkos Tasking: `TaskSingle` and `TaskTeam`.  The former tells Kokkos to launch the associated task functor with a single worker when its predecessors are done (more on this below), while the latter tells Kokkos to launch with a team of workers, similar to a single team from a `Kokkos::TeamPolicy` launch in data parallelism.

Predecessors
------------

Dependency relationships in Kokkos are represented by instances of the `Kokkos::BasicFuture` class template.  Each task (created with the `task_spawn` or `host_spawn` patterns) can have zero or one predecessors.  Predecessors are given to a pattern as an argument to the policy:

```c++
using scheduler_type = /* ... discussed below ... */;
auto scheduler = scheduler_type(/* ... discussed below ... */);
// Launch with no predecessor:
auto fut = Kokkos::host_spawn(
  Kokkos::TaskSingle(scheduler),
  MyTaskFunctor()
);
// Launch when `fut` is ready, at the earliest:
auto fut2 = Kokkos::host_spawn(
  Kokkos::TaskSingle(scheduler, fut),
  MyOtherTaskFunctor()
);
/* ... */
```

Schedulers
----------

The Kokkos `TaskScheduler` concept is an abstraction that generalizes over the many possible strategies for scheduling tasks in a task-based system.  Like other concepts in Kokkos, users should not write code that depends directly on a specific `TaskScheduler`, but rather to the generic model that all `TaskScheduler` types guarantee.

The `Kokkos::BasicFuture` class template, used for representing dependency relationships, is templated on the return type of the task it represents and on the type of the scheduler that was used to execute that task:

```c++
template <class Scheduler>
void my_function(Scheduler sched) {
  // use auto until you need to name the type for some reason
  auto fut = Kokkos::host_spawn(
    Kokkos::TaskSingle(sched),
    MyTaskFunctor()
  );
  /* ... */
  using my_result_type = MyTaskFunctor::value_type;
  // convertibility is guaranteed:
  Kokkos::BasicFuture<my_task_result, Scheduler> ff = fut;
}
```

(Note: Kokkos does not guarantee the specific return type of task parallel patterns, only that they will be convertible to the appropriate `Kokkos::BasicFuture` type.  Use `auto` until you need to name the type for some reason—like storing it in a container, for instance.  Otherwise, Kokkos may be able to provide better performance if the future type is never required to be converted to a specific `Kokkos::BasicFuture` type.)

Future types in Kokkos have shared reference semantics; a copy of a given future represents the same underlying dependency as the future it was copied from.


Invariants in the Kokkos Tasking Programming Model
==================================================

TODO