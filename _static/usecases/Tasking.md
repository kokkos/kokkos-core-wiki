# Kokkos Tasking Use Case

> Deprecated since Kokkos 4.5.

Kokkos provides an execution pattern that enables end-user code to dynamically execute a body of work that 
is not easily prescribed to structured flat or hierarchical parallelism.  This use case describes the characteristics
of an end-user program that is best designed using a tasking execution pattern, and provides examples of program structure
and usage of the Kokkos API.

## Actors
 - Algorithm without predetermined concurrency or parallelism
   - Algorithm where results are accumulative based on recursion
   - Or Algorithm where work is divided dynamically through logical tree
    
## Subjects
 - Kokkos Execution Space
 - Kokkos Task Scheduler
 - Kokkos Task Queue

## Assumptions
 - Number of tasks required cannot be determined before starting
    
## Constraints
 - Number of threads and memory are limited by execution and memory space
    
## Preconditions
 - Task Scheduler and queue must be selected by end-user application
 - Task "kernel" implemented in the form of C++ functor
    
## Usage Pattern
 - Launch Head task either on host or task queue
   - Spawn one or more surrogate tasks
   - wait for surrogate tasks to complete
   - roll-up/accumulate result
 - Surrogate tasks may spawn one or more surrogate tasks
   - wait for spawned tasks to complete
   - roll-up/accumulate result

```
[Start Task Head]
     |-------------- Inside 'head' functor ------------------
     | [Spawn task a]
     |     |------------- Inside 'a' functor ----------------
     |     | *Spawn task a1
     |     | *Spawn task a2
     |     | *Spawn task a3
     |     | Add {a1,a2,a3} to waiting list
     |     | Respawn -- wait until waiting list is complete
     |     |------------ re-enter 'a' functor ---------------
     |     | combine results from {a1,a2,a3} and return
     |     |-------------------------------------------------
     | [Spawn task b]
     |     |------------- Inside 'b' functor ----------------
     |     | *Spawn task b1
     |     | *Spawn task b2
     |     | *Spawn task b3
     |     | Add {b1,b2,b3} to waiting list
     |     | Respawn -- wait until waiting list is complete
     |     |------------ re-enter 'b' functor ---------------
     |     | combine results from {b1,b2,b3} and return
     |     |-------------------------------------------------
     | Add {a,b} to waiting list
     | Respawn -- wait until waiting list is complete
     |-------------- re-enter head functor ------------------
     | combine results from {a,b} and return
     |-------------------------------------------------------
```

## Post conditions
 - Completed tasks return results via Kokkos::future
 - futures of surrogate task (in wait list) are guaranteed to be set when parent functor is re-entered (after respawning)

## Examples

## Recursive Example - Fibonacci Sequence

 - Recursive Algorithm 
   - F<sub>n</sub> = F<sub>n-1</sub> + F<sub>n-2</sub>
   - F<sub>0</sub> = 0 and F<sub>1</sub>=1

### Task Functor

```c++
struct Fib {
  
  using future_type = Kokkos::BasicFuture<return_type, Scheduler>;
  int N = 0;
  future_type f1;
  future_type f2;

  operator() (team_member & member, return_type & return) {
    auto scheduler = member.scheduler();
    if (N < 2) {
        return = N;
    } else if (f1.is_ready() && f2.is_ready()) {
        return = f1.get() + f2.get();
    } else{
        f1 = Kokkos::task_spawn( Kokkos::TaskSingle(scheduler),
                                 Fib{N-1} );
        f2 = Kokkos::task_spawn( Kokkos::TaskSingle(scheduler),
                                 Fib{N-2} );
         Kokkos::BasicFuture<void, Scheduler> wait_list[] = { f1, f2 };
         auto fall = scheduler.when_all(wait_list);
         Kokkos::respawn(this, fall);
    }
  }
 
};
```

### Example flow for N = 3
``` 
[Start head task A(N=3)]
   A_f1 = [Spawn task B N = 2]
   |      |  B_f1 = [Spawn task N = 1]
   |      |  | - return 1
   |      |  B_f2 = [Spawn task N = 0]
   |      |  | - return 0
   |      | - wait for f1 and f2, then respawn
   |      | ----------- re-enter B functor ----------------
   |      | - return (0) + (1)  [result from B_f1 and B_f2]
   A_f2 = [Spawn task C N = 1]
   |      | - return 1
   | - wait for A_f1 and A_f2, then respawn
   | --------- re-enter A functor -------------------------
   | - return (1) + (1)  [result from A_f1 and A_f2] 
```

## Work divided through graph

### Top Down BFS Algorithm

Given league of size LS each, team member TM will pull a vertex off of the search 
queue for that team. Subteam member workers are then spawned to visit each of the
vertices attached to the visited node.  The task is further split if the number of 
vertices exceeds a threshold (256).  When an unvisited (new) node is encountered
then the vertices attached to that node are appended to the team queue.  Work is 
complete when all the queues are empty and the nodes have all been visited.

```
   [Start Task Head]
        |-------------- Inside 'head' functor ------------------------------------------------
        | [Spawn task T = 0]
        |     |------------- Inside 'T=0' functor --------------------------------------------
        |     | *Spawn task TM=0       |
        |     |     ...                |  - Team members added to 
        |     |                        |    wait list
        |     | *Spawn task TM=TS-1    |
        |     |------------------ Inside TM functor ------------------------------------------
        |     | - atomically update frontier queue and retrieve next vertex  
        |     |   |  Spawn Search task ST = 0 - Memory Limit
        |     |   |    |-------------- Inside Search Task functor ----------------------------
        |     |   |    |     | - if edge list from vertex is small, visit each node
        |     |   |    |     | - if edge list is large spawn edge list workers for 
        |     |   |    |     |   every 256 edges
        |     |   |    |     | ---- return after visiting or respawn to wait for edge workers
        |     |   |---------Repeat until queue is empty -------------------------------------
        |     |   | Add each search task to wait queue
down to |     |   |- Respawn -- wait until waiting list is complete
        |     |------------ ------------------------------- ---------------------------------
        |     | Wait for each team member an respawn
        |     |------------------------------------------------------------------------------
        | [Spawn task T = LS-1]
        |     |  (same as above )
        |     |------------------------------------------------------------------------------
        | Add Teams to waiting list
        | Respawn -- wait until waiting list is complete
        |-------------- re-enter head functor -----------------------------------------------
        | combine results from {Teams} and return
        |------------------------------------------------------------------------------------
```

Note that with this algorithm, the queue position, the queue itself, and the data indicating whether
a node has been visited must all be updated atomically.  Thus, the league size will greatly determine
the contention for queue resources.
