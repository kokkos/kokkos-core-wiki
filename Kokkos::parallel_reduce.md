# Kokkos::parallel_reduce()

Header File: `Kokkos_Core.hpp`

### Usage 
```c++
Kokkos::parallel_reduce( name, policy, functor, result);
Kokkos::parallel_reduce( name, policy, functor, reducer);
Kokkos::parallel_reduce( name, policy, functor);
Kokkos::parallel_reduce( policy, functor, result);
Kokkos::parallel_reduce( policy, functor, reducer);
Kokkos::parallel_reduce( policy, functor);
```

Dispatches parallel work defined by `functor` according to the *ExecutionPolicy* `policy` and performance a reduction of the contributions
provided by the work items. The optional label `name` is used by profiling and debugging tools. The reduction type is either a `sum`, 
defined by `reducer` or deduced from an optional `join` operator on the functor. The reduction result is stored in `result`, or through the
`reducer` handle. It is also provided to the `functor.final()` function if such a function exists.  

## Interface

```cpp
template <class ExecPolicy, class FunctorType>
Kokkos::parallel_reduce(const std::string& name, const ExecPolicy& policy, const FunctorType& functor);
```

```cpp
template <class ExecPolicy, class FunctorType>
Kokkos::parallel_reduce(const ExecPolicy& policy, const FunctorType& functor);
```

### Parameters:

  * `name`: A user provided string which is used in profiling and debugging tools via the Kokkos Profiling Hooks. 
  * ExecPolicy: An *ExecutionPolicy* which defines iteration space and other execution properties. Valid policies are:
    * `IntegerType`: defines a 1D iteration range, starting from 0 and going to a count.
    * `RangePolicy`: defines a 1D iteration range. 
    * `MDRangePolicy`: defines a multi-dimensional iteration space.
    * `TeamPolicy`: defines a 1D iteration range, each of which is assigned to a thread team.
    * `TeamThreadRange`: defines a 1D iteration range to be executed by a thread-team. Only valid inside a parallel region executed through a `TeamPolicy` or a `TaskTeam`.
    * `ThreadVectorRange`: defines a 1D iteration range to be executed through vector parallelization. Only valid inside a parallel region executed through a `TeamPolicy` or a `TaskTeam`.
  * FunctorType: A valid functor with an `operator()` with a matching signature for the `ExecPolicy`
  * ReducerArgument is either a class fullfilling the "Reducer" concept, an scalar argument taken by reference or a `Kokkos::View`

### Requirements:
  
  * If `ExecPolicy` is not `MDRangePolicy` the `functor` has a member function of the form `operator() (const HandleType& handle, ReducerValueType& value) const` or `operator() (const WorkTag, const HandleType& handle, ReducerValueType& value) const` 
    * The `WorkTag` free form of the operator is used if `ExecPolicy` is an `IntegerType` or `ExecPolicy::work_tag` is `void`.
    * `HandleType` is an `IntegerType` if `ExecPolicy` is an `IntegerType` else it is `ExecPolicy::member_type`.
  * If `ExecPolicy` is `MDRangePolicy` the `functor` has a member function of the form `operator() (const IntegerType& i0, ... , const IntegerType& iN, ReducerValueType& value) const` or `operator() (const WorkTag, const IntegerType& i0, ... , const IntegerType& iN, ReducerValueType& value) const` 
    * The `WorkTag` free form of the operator is used if `ExecPolicy::work_tag` is not `void`.
    * `N` must match `ExecPolicy::rank`
  * The reduction argument type `ReducerValueType` of the `functor` operator must be compatible with the `ReducerArgument` and must match `init`, `join` and `final` argument types of the functor if those exist.
    * is a scalar type: `ReducerValueType` must be of the same type.
    * is a `Kokkos::View`: `ReducerArgument::rank` must be 1 and `ReducerArgument::non_const_value_type` must match `ReducerValueType`.
    * satisfies the `Reducer` concept: `ReducerArgument::value_type` must match `ReducerValueType`  
  
## Semantics

* Neither concurrency nor order of execution are guaranteed. 
* The call is potentially asynchronous if the `ReducerArgument` is not a scalar type. 
* The `ReducerArgument` content will be overwritten, i.e. the value does not need to be initialized to the reduction-neutral element. 
* The input value to the operator may contain a partial reduction result, Kokkos may only combine the thread local contributions in the end. The operator should modify the input reduction value according to the requested reduction type. 

## Examples

More Detailed Examples are provided in the ExecutionPolicy documentation. 

```c++
#include<Kokkos_Core.hpp>
#include<cstdio> 

int main(int argc, char* argv[]) {
   Kokkos::initialize(argc,argv);

   int N = atoi(argv[1]);
   double result;
   Kokkos::parallel_reduce("Loop1", N, KOKKOS_LAMBDA (const int& i, double& lsum ) {
     lsum += 1.0*i;
   },result);

   printf("Result: %i %lf\n",N,result);
   Kokkos::finalize();
}
```

```c++
#include<Kokkos_Core.hpp>
#include<cstdio> 

struct TagMax {};
struct TagMin {};

struct Foo {
  KOKKOS_INLINE_FUNCTION
  void operator() (const TagMax, const Kokkos::TeamPolicy<>::member_type& team, double& lmax) const {
    if( team.league_rank % 17 + team.team_rank % 13 > lmax )
      lmax = team.league_rank % 17 + team.team_rank % 13;
  });
  KOKKOS_INLINE_FUNCTION
  void operator() (const TagMin, const Kokkos::TeamPolicy<>::member_type& team, double& lmin ) const {
    if( team.league_rank % 17 + team.team_rank % 13 < lmin )
      lmin = team.league_rank % 17 + team.team_rank % 13;
  });
});

int main(int argc, char* argv[]) {
   Kokkos::initialize(argc,argv);

   int N = atoi(argv[1]);

   Foo foo;
   double max,min;
   Kokkos::parallel_reduce(Kokkos::TeamPolicy<TagMax>(N,Kokkos::AUTO), foo, Kokkos::Max<double>(max));
   Kokkos::parallel_reduce("Loop2", Kokkos::TeamPolicy<TagMin>(N,Kokkos::AUTO), foo, Kokkos::Min<double>(min));
   Kokkos::fence();

   printf("Result: %lf %lf\n",min,max);

   Kokkos::finalize();
}
```


