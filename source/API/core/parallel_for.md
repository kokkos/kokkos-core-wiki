# `parallel_for()`

Header File: `Kokkos_Core.hpp`

Usage: 
```c++
Kokkos::parallel_for(name, policy, functor);
Kokkos::parallel_for(policy, functor);
```

Dispatches parallel work defined by `functor` according to the *ExecutionPolicy* `policy`. The optional label `name` is
used by profiling and debugging tools. This call may be asynchronous and return to the callee immediately. 

## Interface

```cpp
template <class ExecPolicy, class FunctorType>
Kokkos::parallel_for(const std::string& name, 
                     const ExecPolicy& policy, 
                     const FunctorType& functor);
```

```cpp
template <class ExecPolicy, class FunctorType>
Kokkos::parallel_for(const ExecPolicy& policy, 
                     const FunctorType& functor);
```

### Parameters:

  * `name`: A user provided string which is used in profiling and debugging tools via the Kokkos Profiling Hooks. 
  * ExecPolicy: An *ExecutionPolicy* which defines iteration space and other execution properties. Valid policies are:
    * `IntegerType`: defines a 1D iteration range, starting from 0 and going to a count.
    * [RangePolicy](Kokkos%3A%3ARangePolicy): defines a 1D iteration range. 
    * [MDRangePolicy](Kokkos%3A%3AMDRangePolicy): defines a multi-dimensional iteration space.
    * [TeamPolicy](Kokkos%3A%3ATeamPolicy): defines a 1D iteration range, each of which is assigned to a thread team.
    * [TeamThreadRange](Kokkos%3A%3ANestedPolicies): defines a 1D iteration range to be executed by a thread-team. Only valid inside a parallel region executed through a `TeamPolicy` or a `TaskTeam`.
    * [ThreadVectorRange](Kokkos%3A%3ANestedPolicies): defines a 1D iteration range to be executed through vector parallelization dividing the threads within a team.  Only valid inside a parallel region executed through a `TeamPolicy` or a `TaskTeam`.
  * FunctorType: A valid functor having an `operator()` with a matching signature for the `ExecPolicy`.  The functor can be defined using a C++ class/struct or lambda.  See Examples below for more detail.

### Requirements:
  
  * If `ExecPolicy` is an `IntegerType`, `functor` has a member function `operator() (const IntegerType& i) const`.  
  * If `ExecPolicy` is an `MDRangePolicy` and `ExecPolicy::work_tag` is `void`, `functor` has a member function `operator() (const IntegerType& i0, ... , const IntegerType& iN) const` where `N` is `ExecPolicy::rank-1`.
  * If `ExecPolicy` is an `MDRangePolicy` and `ExecPolicy::work_tag` is not `void`, `functor` has a member function `operator() (const ExecPolicy::work_tag, const IntegerType& i0, ... , const IntegerType& iN) const` where `N` is `ExecPolicy::rank-1`.
  * If `ExecPolicy::work_tag` is `void`, `functor` has a member function `operator() (const ExecPolicy::member_type& handle) const`.
  * If `ExecPolicy::work_tag` is not `void`, `functor` has a member function `operator() (const ExecPolicy::work_tag, const ExecPolicy::member_type& handle) const`. 

## Semantics

* Neither concurrency nor order of execution of iterations are guaranteed.
* The call is potentially asynchronous. To guarantee a kernel has finished, a developer should call fence on the execution space on which the kernel is run.

## Examples

More Detailed Examples are provided in the ExecutionPolicy documentation. 

* `IntegerType` policy with lambda as the functor.  Note that KOKKOS_LAMBDA is the same as [=] KOKKOS_FUNCTION, which means that all of the variables used within the lambda are captured by value.  Also, the KOKKOS_LAMBDA and KOKKOS_FUNCTION macros add all of the function specifiers necessary for the target execution space.

```c++
#include<Kokkos_Core.hpp>
#include<cstdio> 

int main(int argc, char* argv[]) {
   Kokkos::initialize(argc,argv);

   int N = atoi(argv[1]);

   Kokkos::parallel_for("Loop1", N, KOKKOS_LAMBDA (const int& i) {
     printf("Greeting from iteration %i\n",i);
   });

   Kokkos::finalize();
}
```

* `TeamPolicy` policy with C++ struct as  functor.  Note that the KOKKOS_INLINE_FUNCTION macro adds all of the function specifiers necessary for the target execution space.  The TagA/B structs also provide the ability to 'overload' the operators within the same functor.  Much like the lambda example, the functor and any member variables contained within are captured by value, which means they must have either implicit or explicit copy constructors.

```c++
#include<Kokkos_Core.hpp>
#include<cstdio> 

struct TagA {};
struct TagB {};

struct Foo {
  KOKKOS_INLINE_FUNCTION
  void operator() (const TagA, const Kokkos::TeamPolicy<>::member_type& team) const {
    printf("Greetings from thread %i of team %i with TagA\n",
            team.thread_rank(),team.league_rank());
  }
  KOKKOS_INLINE_FUNCTION
  void operator() (const TagB, const Kokkos::TeamPolicy<>::member_type& team) const {
    printf("Greetings from thread %i of team %i with TagB\n",
            team.thread_rank(),team.league_rank());
  }
};

int main(int argc, char* argv[]) {
   Kokkos::initialize(argc,argv);

   int N = atoi(argv[1]);

   Foo foo;

   Kokkos::parallel_for(Kokkos::TeamPolicy<TagA>(N,Kokkos::AUTO), foo);
   Kokkos::parallel_for("Loop2", Kokkos::TeamPolicy<TagB>(N,Kokkos::AUTO), foo);
   
   Kokkos::finalize();
}
```
