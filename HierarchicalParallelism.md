# Chapter 8

# Hierarchical Parallelism

This chapter explains how to use Kokkos to exploit multiple levels of shared-memory parallelism.
These levels include thread teams, threads within a team, and vector lanes.
You may nest these levels of parallelism,
and execute \lstinline!parallel_for!, \lstinline!parallel_reduce!, or \lstinline!parallel_scan! at each level.
The syntax differs only by the execution policy,
which is the first argument to the \verb!parallel_*! operation.
Kokkos also exposes a ``scratch pad'' memory which provides thread private and team private allocations..

## 8.1 Motivation

Node architectures on modern high-performance computers are characterized by ever more \emph{hierarchical parallelism}.
A level in the hierachy is determined by the hardware resources which are shared between compute units at that level.
Higher levels in the hierarchy also have access to all resources in its branch at lower levels of the hierarchy.
This concept is orthogonal to the concept of heterogeneity.
For example, a node in a typical CPU-based cluster consists of a number of multicore CPUs.  Each core supports one or more hyperthreads, and each hyperthread can execute vector instructions.
This means there are 4 levels in the hierarchy of parallelism:
\begin{enumerate}
\item CPU sockets share access to the same memory and network resources,
\item cores within a socket typically have a shared last level cache (LLC),
\item hyperthreads on the same core have access to a shared L1 (and L2) cache and they submit instructions to the same execution units, and
\item vector units execute a shared instruction on multiple data items.
\end{enumerate}
GPU-based systems also have a hierarchy of 4 levels:
\begin{enumerate}
\item Multiple GPUs in the same node share access to the same host memory and network resources,
\item core clusters (e.g. the SMs on an NVIDIA GPU) have a shared cache and access to the same high bandwidth memory on a single GPU,
\item threads running on the same core cluster have access to the same L1 cache and scratch memory and
\item they are grouped in so called Warps or Wave Fronts within which threads are always synchronous and can collaborate more closely, for example via direct register swapping.
\end{enumerate}
Kokkos provides a number of abstract levels of parallelism,
which it maps to the appropriate hardware features.
This mapping is not necessarily static or predefined; it may differ for each kernel.
Furthermore, some mapping decisions happen at run time.
This enables adaptive kernels which map work to different hardware resources depending on the work set size.
While Kokkos provides defaults and suggestions, the optimal mapping can be algorithm dependent.
Hierarchical parallelism is accessible through execution policies.

You should use Hierarchical Parallelism in particular in a number of cases:
\begin{enumerate}
\item Non-tightly nested loops: Hierarchical Parallelism allows you to expose more parallelism.
\item Data gather + reuse: If you gather data for a particular iteration of an outer loop, and then repeatably use it in an inner loop, Hierarchical Parallelism with scratch memory may match the usecase well.
\item Force Cache Blocking: Using Hierarchical Parallelism forces a developer into algorithmic choices which are good for cache blocking. This can sometimes lead to better performing algorithms than a simple flat parallelism.
\end{enumerate}

On the other hand you should probably not use Hierarchical Parallelism if you have tightly nested loops.
For that usecase a multi dimensional RangePolicy is the better fit.

## 8.2 Thread teams

Kokkos' most basic hierarchical parallelism concept is a thread team.
A \emph{thread team} is a collection of threads which can synchronize,
and which share a ``scratch pad'' memory
(see Section\ref {S:Hierarchical:Scratch}).

Instead of mapping a 1-D range of indices to hardware resources,
Kokkos' thread teams map a 2-D index range.
The first index is the \emph{league rank}, the index of the team.
The second index is the \emph{team rank}, the thread index within a team.
In CUDA this is equivalent to launching a 1-D grid of 1-D blocks.
The league size is arbitrary -- that is, it is only limited by the integer size type -- while the team size must fit in the hardware constraints.
As in CUDA, only a limited number of teams are actually active at the same time,
and they must run to completion before new ones are executed.
Consequently it is not valid to use inter thread-team synchronization mechanisms
such as waits for events initiated by other thread teams.

### 8.2.1 Creating a Policy instance

Kokkos exposes use of thread teams with the \lstinline!Kokkos::TeamPolicy! execution policy.
To use thread teams you need to create a \lstinline|Kokkos::TeamPolicy| instance.
It can be created inline for the parallel dispatch call.
The constructors require two arguments: a league size and a team size.
In place of the team size a user can utilize \lstinline|Kokkos::AUTO| to let Kokkos guess a good team size for a given architecture.
Doing that is the recommend way for most developers to utilize the \lstinline|TeamPolicy|.
As with the  \lstinline|Kokkos::RangePolicy| a specific execution tag, a specific execution space, a \lstinline|Kokkos::IndexType|, and a \lstinline|Kokkos::Schedule| can be given as optional template arguments.
\begin{lstlisting}
// Using default execution space and launching
// a league with league_size teams with team_size threads each
Kokkos::TeamPolicy<>
        policy( league_size, team_size );

// Using  a specific execution space to
// run an n_workset parallelism with Kokkos choosing the team size
Kokkos::TeamPolicy<ExecutionSpace>
        policy( league_size, Kokkos::AUTO() );

// Using a specific execution space and an execution tag
Kokkos::TeamPolicy<SomeTag, ExecutionSpace>
        policy( league_size, team_size );
\end{lstlisting}


### 8.2.2 Basic kernels

The team policy's \lstinline!member_type! provides the necessary functionality to use teams within a parallel kernel.
It allows access to thread identifiers such as the league rank and size, and the team rank and size.
It also provides team-synchronous actions such as team barriers, reductions and scans.
\begin{lstlisting}
using Kokkos::TeamPolicy;
using Kokkos::parallel_for;

typedef TeamPolicy<ExecutionSpace>::member_type member_type;
// Create an instance of the policy
TeamPolicy<ExecutionSpace> policy (league_size, Kokkos::AUTO() );
// Launch a kernel
parallel_for (policy, KOKKOS_LAMBDA (member_type team_member) {
    // Calculate a global thread id
    int k = team_member.league_rank () * team_member.team_size () +
            team_member.team_rank ();
    // Calculate the sum of the global thread ids of this team
    int team_sum = team_member.reduce (k);
    // Atomicly add the value to a global value
    a() += team_sum;
  });
\end{lstlisting}

The name ``\lstinline!TeamPolicy!'' makes it explicit that a kernel
using it constitutes a parallel region with respect to the team.

## 8.3 Team scratch pad memory

Each Kokkos team has a ``scratch pad.''
This is an instance of a memory space accessible only by threads in that team.
Scratch pads let an algorithm load a workset into a shared space
and then collaboratively work on it with all members of a team.
The lifetime of data in a scratch pad is the lifetime of the team.
In particular, scratch pads are recycled by all logical teams running on the same physical set of cores.
During the lifetime of the team all operations allowed on global memory are allowed on the scratch memory.
This includes taking addresses and performing atomic operations on elements located in scratch space.
Team-level scratch pads correspond to the per-block shared memory in Cuda,
or to the ``local store'' memory on the Cell processor.

Kokkos exposes scratch pads through a special memory space associated with the execution space:
\lstinline|execution_space::scratch_memory_space|.
You may allocate a chunk of scratch memory through the \lstinline|TeamPolicy| member type.
You may request multiple allocations from scratch, up to a user-provided maximum aggregate size.
The maximum is provided either through a \lstinline|team_shmem_size| function in the functor which returns a potentially team-size dependent value,
or it can be specified through a setting of the TeamPolicy \lstinline|set_scratch_size|.
It is not valid to provide both values at the same time.
The argument to the TeamPolicy can be used to set the shared memory size when using functors.
One restriction on shared memory allocations is that they can not be freed during the lifetime of the team.
This avoids the complexity of a memory pool,
and reduces the time it takes to obtain an allocation
(which currently is a few tens of integer operations to calculate the offset).

The following is an example of using the functor interface:
\begin{lstlisting}
template<class ExecutionSpace>
struct functor {
  typedef ExecutionSpace execution_space;
  typedef execution_space::member_type member_type;

  KOKKOS_INLINE_FUNCTION
  void operator() (member_type team_member) const {
    size_t double_size = 5*team_member.team_size()*sizeof(double);

    // Get a shared team allocation on the scratch pad
    double* team_shared_a = (double*)
      team_member.team_shmem().get_shmem(double_size);

    // Get another allocation on the scratch pad
    int* team_shared_b = (int*)
      team_member.team_shmem().get_shmem(160*sizeof(int));

    // ... use the scratch allocations ...
  }

  // Provide the shared memory capacity.
  // This function takes the team_size as an argument,
  // which allows team_size dependent allocations.
  size_t team_shmem_size (int team_size) const {
    return sizeof(double)*5*team_size +
           sizeof(int)*160;
  }
};
\end{lstlisting}

The \lstinline|set_scratch_size| function of the \lstinline|TeamPolicy| takes two or three arguments.
The first argument specifies the level in the scratch hierarchy for which a specific size is requested.
Different levels have different restrictions.
Generally the first level is restricted to a few tenths of kilobyte roughly corresponding to L1 cache size.
The second level can be used to get an aggregate over all teams of a few gigabyte, corresponding to available
space in high-bandwidth memory.
The third level generally falls back to capacity memory in the node.
The second and third argument are either per-thread or per-team sizes for scratch memory.
Note like previously discussed, the setter function does not modify the instance it is called on, but returns
a copy of the policy object with adjusted scratch size request.

Here are some examples:
\begin{lstlisting}
TeamPolicy<> policy_1 = TeamPolicy<>(league_size, team_size).
                          set_scratch_size(1, PerTeam(1024), PerThread(32));
TeamPolicy<> policy_2 = TeamPolicy<>(league_size, team_size).
                          set_scratch_size(1, PerThread(32));
TeamPolicy<> policy_3 = TeamPolicy<>(league_size, team_size).
                          set_scratch_size(1, PerTeam(1024));
\end{lstlisting}

The total amount of scratch space available for each team will be the per-team value plus the per-thread value multiplied by the team-size.
The interface allows users to specify those settings inline:
\begin{lstlisting}
parallel_for(TeamPolicy<>(league_size, team_size).set_scratch_size(1, PerTeam(1024)),
  KOKKOS_LAMBDA (const TeamPolicy<>::member_type& team) {
    ...
});
\end{lstlisting}

Instead of simply getting raw allocations in memory, users can also allocate Views directly in scratch memory.
This is achieved by providing the shared memory handle as the first argument of the View constructor.
Views also have a static member function which return their shared memory size requirements.
The function expects the run-time dimensions as arguments, corresponding to View's constructor.
Note that the view must be unmanaged (i.e. it must have the \lstinline|Unmanaged| memory trait).

\begin{lstlisting}
tyepdef Kokkos::DefaultExecutionSpace::scratch_memory_space
  ScratchSpace;
// Define a view type in ScratchSpace
typedef Kokkos::View<int*[4],ScratchSpace,
          Kokkos::MemoryTraits<Kokkos::Unmanaged>> shared_int_2d;

// Get the size of the shared memory allocation
size_t shared_size = shared_int_2d::shmem_size(team_size);
Kokkos::parallel_for(Kokkos::TeamPolicy<>(league_size,team_size),
                     KOKKOS_LAMBDA ( member_type team_member) {
  // Get a view allocated in team shared memory.
  // The constructor takes the shared memory handle and the
  // runtime dimensions
  shared_int_2d A(team_member.team_shmem(), team_member.team_size());
  ...
});
\end{lstlisting}

## 8.4 Nested parallelism

Instead of writing code which explicitly uses league and team rank indices, one can use nested parallelism to implement hierarchical algorithms.
Kokkos lets the user have up to three nested layers of parallelism.
The team and thread levels are the first two levels.
The third level is \emph{vector} parallelism.

You may use any of the three parallel patterns -- for, reduce, or scan -- at each level\footnote{The parallel scan operation is not implemented for all execution spaces on the thread level, and it doesn't support a TeamPolicy on the top level.}.
You may nest them and use them in conjunction with code that is aware of the league and team rank.
The different layers are accessible via special execution policies:
\lstinline|TeamThreadLoop| and \lstinline|ThreadVectorLoop|.

### 8.4.1 Team loops

The first nested level of parallel loops splits an index range over the threads of a team.
This motivates the policy name \lstinline|TeamThreadRange|,
which indicates that the loop is executed once by the team with the index range split over threads.
The loop count is not limited to the number of threads in a team, and how the index range is mapped to threads is architecture dependent.
It is not legal to nest multiple parallel loops using the \lstinline!TeamThreadRange! policy.
However, it is valid to have multiple parallel loops using the \lstinline!TeamThreadRange! policy follow each other in sequence, in the same kernel.
Note that it is not legal to make a write access to POD data outside of the closure of a nested parallel layer. 
This is a conscious choice to prevent difficult to debug issues related to thread private, team shared and globally shared variables.
A simple way to enforce this is by using the ``capture by value'' clause with lambdas,
but ``capture by reference'' is recommended for release builds since it typically results in better performance.
With the lambda being considered as \lstinline|const| inside the \lstinline!TeamThreadRange! loop,
the compiler will catch illegal accesses at compile time as a \lstinline|const| violation.

The simplest use case is to have another \lstinline|parallel_for| nested inside a kernel.
\begin{lstlisting}
using Kokkos::parallel_for;
using Kokkos::TeamPolicy;
using Kokkos::TeamThreadRange;

parallel_for (TeamPolicy<> (league_size, team_size),
                    KOKKOS_LAMBDA (member_type team_member)
{
  Scalar tmp;
  parallel_for (TeamThreadRange (team_member, loop_count),
    [=] (int& i) {
      // ...
      // tmp += i; // This would be an illegal access
    });
});
\end{lstlisting}

The \lstinline|parallel_reduce| construct can be used to perform optimized team-level reductions:

\begin{lstlisting}
using Kokkos::parallel_reduce;
using Kokkos::TeamPolicy;
using Kokkos::TeamThreadRange;
parallel_for (TeamPolicy<> (league_size, team_size),
                 KOKKOS_LAMBDA (member_type team_member) {
    // The default reduction uses Scalar's += operator
    // to combine thread contributions.
    Scalar sum;
    parallel_reduce (TeamThreadRange (team_member, loop_count),
      [=] (int& i, Scalar& lsum) {
        // ...
        lsum += ...;
      }, sum);

    // You may provide a custom reduction as another
    // lambda together with an initialization value.
    Scalar product;
    Scalar init_value = 1;
    parallel_reduce (TeamThreadRange (team_member, loop_count),
      [=] (int& i, Scalar& lsum) {
        // ...
        lsum *= ...;
      }, product, [=] (Scalar& lsum, Scalar& update) {
        lsum *= update;
      }, init_value);
  });
\end{lstlisting}

The third pattern is \lstinline|parallel_scan| which can be used to perform prefix scans.

### 8.4.2 Vector loops

At the innermost level of nesting parallel loops in a kernel is comprised of the \emph{vector}-loop.
Vector level parallelism works identical to the team level loops using the execution policy \lstinline|ThreadVectorRange|.
In contrast to the team-level, there is no legal way to exploit the vector level outside of a parallel pattern using the \lstinline|ThreadVectorRange|.
However one can use such a parallel construct in- and outside- of a \lstinline|TeamThreadRange| parallel operation.

\begin{lstlisting}
using Kokkos::parallel_reduce;
using Kokkos::TeamPolicy;
using Kokkos::TeamThreadRange;
using Kokkos::ThreadVectorRange;
parallel_for (TeamPolicy<> (league_size, team_size),
                 KOKKOS_LAMBDA (member_type team_member) {

    int k = team_member.team_rank();
    // The default reduction uses Scalar's += operator
    // to combine thread contributions.
    Scalar sum;
    parallel_reduce (ThreadVectorRange (team_member, loop_count),
      [=] (int& i, Scalar& lsum) {
        // ...
        lsum += ...;
      }, sum);

    parallel_for (TeamThreadRange (team_member, workset_size),
      [&] (int& j) {
      // You may provide a custom reduction as another
      // lambda together with an initialization value.
      Scalar product;
      Scalar init_value = 1;
      parallel_reduce (ThreadVectorRange (team_member, loop_count),
        [=] (int& i, Scalar& lsum) {
          // ...
          lsum *= ...;
        }, product, [=] (Scalar& lsum, Scalar& update) {
          lsum *= update;
        }, init_value);
      });
  });
\end{lstlisting}

As the name indicates the vector-level must be vectorizable.
The parallel patterns will exploit available mechanisms to encourage vectorization by the compiler.
When using the Intel compiler for example, the vector level loop will be internally decorated with
\lstinline|#pragma ivdep|, telling the compiler to ignore assumed vector dependencies.

### 8.4.3 Restricting execution to a single executor

As stated above, a kernel is a parallel region with respect to threads (and vector lanes) within a team.
This means that global memory accesses outside of the respective nested levels potentially have to be protected against repetitive execution.
A common example is the case where a team performs some calculation but only one result per team has to be written back to global memory.

Kokkos provides the \lstinline|Kokkkos::single(Policy,Lambda)| function for this case.
It currently accepts two policies:
\begin{itemize}
\item \lstinline|Kokkos::PerTeam| restricts execution of the lambda's
  body to once per team
\item \lstinline|Kokkos::PerThread| restricts execution of the
  lambda's body to once per thread (that is, to only one vector lane
  in a thread)
\end{itemize}
The \lstinline|single| function takes a lambda as its second argument.
That lambda takes zero arguments or one argument by reference.
If it takes no argument, its body must perform side effects in order to have an effect.
If it takes one argument, the final value of that argument is broadcast to every executor on the level:
i.e. every vectorlane of the thread, or every thread (and vector lane) of the team.
It must always be correct for the lambda to capture variables by value
(\lstinline|[=]|, not \lstinline|[&]|).
Thus, if the lambda captures by reference,
it must \emph{not} modify variables that it has captured by reference.

\begin{lstlisting}
using Kokkos::parallel_for;
using Kokkos::parallel_reduce;
using Kokkos::TeamThreadRange;
using Kokkos::ThreadVectorRange;
using Kokkos::PerThread;

TeamPolicy<...> policy (...);
typedef TeamPolicy<...>::member_type team_member;

parallel_for (policy, KOKKOS_LAMBDA (const team_member& thread) {
  // ...

  parallel_for (TeamThreadRange (thread, 100),
    KOKKOS_LAMBDA (const int& i) {
      double sum = 0;
      // Perform a vector reduction with a thread
      parallel_reduce (ThreadVectorRange (thread, 100),
        [=] (int i, double& lsum) {
          // ...
          lsum += ...;
      }, sum);
      // Add the result value into a team shared array.
      // Make sure it is only added once per thread.
      Kokkos::single (PerThread (), [=] () {
          shared_array(i) += sum;
      });
  });

  double sum;
  parallel_reduce (TeamThreadRange (thread, 99),
    KOKKOS_LAMBDA (int i, double& lsum) {
      // Add the result value into a team shared array.
      // Make sure it is only added once per thread.
      Kokkos::single (PerThread (thread), [=] () {
          lsum += someFunction (shared_array(i),
                                shared_array(i+1));
      });
  }, sum);

  // Add the per team contribution to global memory.
  Kokkos::single (PerTeam (thread), [=] () {
    global_array(thread.league_rank()) = sum;
  });
});
\end{lstlisting}

Here is an example of using the broadcast capabilities to determine the start offset for a team
in a buffer:

\begin{lstlisting}
using Kokkos::parallel_for;
using Kokkos::parallel_reduce;
using Kokkos::TeamThreadRange;
using Kokkos::ThreadVectorRange;
using Kokkos::PerThread;

TeamPolicy<...> policy (...);
typedef TeamPolicy<...>::member_type team_member;

Kokkos::View<int> offset("Offset");
offset() = 0;

parallel_for (policy, KOKKOS_LAMBDA (const team_member& thread) {
  // ...

  parallel_reduce (TeamThreadRange (thread, 100),
    KOKKOS_LAMBDA (const int& i, int& lsum) {
      if(...) lsum++;
  });
  Kokkos::single (PerTeam (thread), [=] (int& my_offset) {
    my_offset = Kokkos::atomic_fetch_add(&offset(),lsum);
  });
  ...
\end{lstlisting}

To further illustrate the "parallel region" semantics of the team execution consider the following code:

\begin{lstlisting}
using Kokkos::parallel_reduce;
using Kokkos::TeamThreadRange;
using Kokkos::TeamPolicy;

parallel_reduce(TeamPolicy<>(N,team_size),
  KOKKOS_LAMBDA (const member_type& teamMember, int& lsum) {
    int s = 0;
    for(int i = 0; i<10; i++) s++;
    lsum += s;
},sum);
\end{lstlisting}

In this example \lstinline{sum} will contain the value \lstinline{N * team_size * 10}.
Every thread in each team will compute \lstinline{s=10} and then contribute it to the sum.

Lets go one step further and add a nested \lstinline{parallel_reduce}.
By choosing the loopbound to be \lstinline{team_size} every thread still only runs
once through the inner loop.

\begin{lstlisting}
using Kokkos::parallel_reduce;
using Kokkos::TeamThreadRange;
using Kokkos::TeamPolicy;

parallel_reduce(TeamPolicy<>(N,team_size),
  KOKKOS_LAMBDA (const member_type& teamMember, int& lsum) {

  int s = 0;
  parallel_reduce(TeamThreadRange(teamMember, team_size),
    [=] (const int k, int & inner_lsum) {
    int inner_s = 0;
    for(int i = 0; i<10; i++) inner_s++;
    inner_lsum += inner_s;
  },s);
  lsum += s;
},sum);
\end{lstlisting}

The answer in this case is neverless \lstinline{N * team_size * team_size * 10}.
Each thread computes \lstinline{inner_s = 10}.
But all threads in the team combine their results to compute a \lstinline{s} value of \lstinline{team_size * 10}.
Since every thread in each team contributes that value to the global sum, we arrive at the final value of \lstinline{N * team_size * team_size * 10}.
If the intended goal was for each team to only contribute \lstinline{s} once to the global sum,
the contribution should have been protected with a \lstinline{single} clause.
