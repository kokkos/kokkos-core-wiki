# Kokkos::parallel_scan()

Header File: `Kokkos_Core.hpp`

### Usage 
```c++
Kokkos::parallel_scan( name, policy, functor, result );
Kokkos::parallel_scan( name, policy, functor );
Kokkos::parallel_scan( policy, functor, result);
Kokkos::parallel_scan( policy, functor );
```

Dispatches parallel work defined by `functor` according to the *ExecutionPolicy* `policy` and perform a pre or post scan of the contributions
provided by the work items. The optional label `name` is used by profiling and debugging tools.  If provided, the final result is placed in result. 

## Interface

```cpp
template <class ExecPolicy, class FunctorType>
Kokkos::parallel_scan(const std::string& name, 
                      const ExecPolicy& policy, 
                      const FunctorType& functor);
```

```cpp
template <class ExecPolicy, class FunctorType>
Kokkos::parallel_scan(const ExecPolicy&  policy, 
                      const FunctorType& functor);
```

```cpp
template <class ExecPolicy, class FunctorType, class ReturnType>
Kokkos::parallel_scan(const std::string& name, 
                      const ExecPolicy&  policy, 
                      const FunctorType& functor, 
                      ReturnType&        return_value);
```

```cpp
template <class ExecPolicy, class FunctorType, class ReturnType>
Kokkos::parallel_scan(const ExecPolicy&  policy, 
                      const FunctorType& functor, 
                      ReturnType&        return_value);
```

```cpp
template <class ExecPolicy, class FunctorType, class ReturnType>
Kokkos::parallel_scan(const std::string& name, 
                      const ExecPolicy&  policy, 
                      const FunctorType& functor, 
                      ReturnType&        return_value);
```

```cpp
template <class ExecPolicy, class FunctorType, class ReducerArgumentNonConst>
Kokkos::parallel_reduce(const ExecPolicy& policy, 
                        const FunctorType& functor, 
                        ReducerArgumentNonConst& reducer);
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
  * FunctorType: A valid functor with (at minimum) an `operator()` with a matching signature for the `ExecPolicy` combined with the reduced type.
  * ReturnType: a POD type with `operator +=` and `operator =`, or a `Kokkos::View`.  

### Requirements:
  
  * If `ExecPolicy` is not `MDRangePolicy` the `functor` has a member function of the form `operator() (const HandleType& handle, ReturnType& value, const bool final) const` or `operator() (const WorkTag, const HandleType& handle, ReturnType& value, const bool final) const` 
    * The `WorkTag` free form of the operator is used if `ExecPolicy` is an `IntegerType` or `ExecPolicy::work_tag` is `void`.
    * `HandleType` is an `IntegerType` if `ExecPolicy` is an `IntegerType` else it is `ExecPolicy::member_type`.
  * If `ExecPolicy` is `MDRangePolicy` the `functor` has a member function of the form `operator() (const IntegerType& i0, ... , const IntegerType& iN, ReturnType& value, const bool final) const` or `operator() (const WorkTag, const IntegerType& i0, ... , const IntegerType& iN, ReturnType& value, const bool final) const` 
    * The `WorkTag` free form of the operator is used if `ExecPolicy::work_tag` is not `void`.
    * `N` must match `ExecPolicy::rank`
  * The type `ReturnType` of the `functor` operator must be compatible with the `ReturnType` of the parallel_scan and must match the arguments of the `init` and `join` functions of the functor.  
  * the functor must define FunctorType::value_type the same as ReturnType
       
## Semantics

* Neither concurrency nor order of execution are guaranteed. 
* The `ReturnType` content will be overwritten, i.e. the value does not need to be initialized to the reduction-neutral element. 
* The input value to the operator may contain a partial result, Kokkos may only combine the thread local contributions in the end. The operator should modify the input value according to the desired scan operation. 

## Examples

```c++
#include<Kokkos_Core.hpp>
#include<cstdio> 

int main(int argc, char* argv[]) {
   Kokkos::initialize(argc,argv);

   int N = atoi(argv[1]);
   double result;
   ScanFunctor f;
   Kokkos::parallel_scan("Loop1", N, f, result);

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
   Kokkos::parallel_scan(Kokkos::TeamPolicy<TagMax>(N,Kokkos::AUTO), foo, max);
   Kokkos::parallel_scan("Loop2", Kokkos::TeamPolicy<TagMin>(N,Kokkos::AUTO), foo, min);
   Kokkos::fence();

   printf("Result: %lf %lf\n",min,max);

   Kokkos::finalize();
}
```


