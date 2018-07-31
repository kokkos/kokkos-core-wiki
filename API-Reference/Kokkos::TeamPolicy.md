# `Kokkos::TeamPolicy`

Header File: `Kokkos_Core.hpp`

Usage: 
  ```c++
  Kokkos::TeamPolicy<>( league_size, team_size [, vector_length])
  Kokkos::TeamPolicy<ARGS>(league_size, team_size [, vector_length])
  Kokkos::TeamPolicy<>(Space, league_size, team_size [, vector_length])
  Kokkos::TeamPolicy<ARGS>(Space, league_size, team_size [, vector_length])
  ```

TeamPolicy defines an execution policy for a 1D iteration space starting at begin and going to end with an open interval. 

See also: [TeamMember](Kokkos%3A%3ATeamHandleConcept)

# Synopsis 
  ```c++
  template<class ... Args>
  class Kokkos::TeamPolicy {
    typedef TeamPolicy execution_policy;

    //Inherited from PolicyTraits<Args...>  
    using execution_space   = PolicyTraits<Args...>::execution_space; 
    using schedule_type     = PolicyTraits<Args...>::schedule_type; 
    using work_tag          = PolicyTraits<Args...>::work_tag; 
    using index_type        = PolicyTraits<Args...>::index_type; 
    using iteration_pattern = PolicyTraits<Args...>::iteration_pattern; 
    using launch_bounds     = PolicyTraits<Args...>::launch_bounds;

    typedef TeamMemberType<execution_space> member_type ;

    //Constructors
    TeamPolicy(const TeamPolicy&) = default;
    TeamPolicy(TeamPolicy&&) = default;

    inline TeamPolicy();

    template<class ... Args>
    inline TeamPolicy( const typename traits::execution_space & work_space
             , const index_type league_size
             , const index_type team_size
             , const index_type vector_length = 1);

    template<class ... Args>
    inline TeamPolicy( const index_type league_size
             , const index_type team_size
             , const index_type vector_length = 1);

    TeamPolicy& operator = (const TeamPolicy&) = default;

    
    // set chunk_size to a discrete value
    inline TeamPolicy& set_chunk_size(int chunk);
    // set scratch size for per-team and/or per-thread
    inline TeamPolicy& set_scratch_size(const int& level, const Impl::PerTeamValue& per_team);
    inline TeamPolicy& set_scratch_size(const int& level, const Impl::PerThreadValue& per_thread);
    inline TeamPolicy& set_scratch_size(const int& level, const Impl::PerTeamValue& per_team, const Impl::PerThreadValue& per_thread);
    inline TeamPolicy& set_scratch_size(const int& level, const Impl::PerThreadValue& per_thread, const Impl::PerTeamValue& per_team);

    // return ExecSpace instance provided to the constructor
    KOKKOS_INLINE_FUNCTION const typename traits::execution_space & space() const;
  };
  ```

## Parameters:

### Common Arguments for all Execution Policies

  * Execution Policies generally accept compile time arguments via template parameters and runtime parameters via constructor arguments or setter functions.
  * Template arguments can be given in arbitrary order.

| Argument | Options | Purpose |
| --- | --- | --- |
| ExecutionSpace | `Serial`, `OpenMP`, `Threads`, `Cuda`, `ROCm` | Specify the Execution Space to execute the kernel in. Defaults to `Kokkos::DefaultExecutionSpace`. |
| Schedule | `Schedule<Dynamic>`, `Schedule<Static>` | Specify scheduling policy for work items. `Dynamic` scheduling is implemented through a work stealing queue. Default is machine and backend specific. |
| IndexType | `IndexType<int>` | Specify integer type to be used for traversing the iteration space. Defaults to `int64_t`. |
| WorkTag | `SomeClass` | Specify the work tag type used to call the functor operator. Any arbitrary type defaults to `void`. |

### Requirements:


## Public Class Members

### Constructors
 
 * ```c++
   TeamPolicy()
   ```
   Default Constructor uninitialized policy.
 
 * ```c++
   TeamPolicy(index_type league_size, index_type team_size [, index_type vector_length])
   ```
   Request to launch `league_size` work items, each of which is assigned to a team of threads with `team_size` threads, using a vector length of `vector_length`.
   If the team size is not possible when calling a parallel policy, that kernel launch may throw. 
 
 * ```c++
   TeamPolicy(index_type league_size, Impl::AUTO_t [, index_type vector_length])
   ```
   Request to launch `league_size` work items, each of which is assigned to a team of threads of a size determined by Kokkos, using a vector length of `vector_length`.
   The team size may be determined lazily at launch time, taking into account properties of the functor.

 * ```c++
   TeamPolicy(execution space, index_type league_size, index_type team_size [, index_type vector_length])
   ```
   Request to launch `league_size` work items, each of which is assigned to a team of threads with `team_size` threads, using a vector length of `vector_length`.
   If the team size is not possible when calling a parallel policy, that kernel launch may throw. 
   Use the provided execution space instance during a kernel launch.  

 * ```c++
   TeamPolicy(execution_space space, index_type league_size, Impl::AUTO_t [, index_type vector_length])
   ```
   Request to launch `league_size` work items, each of which is assigned to a team of threads of a size determined by Kokkos, using a vector length of `vector_length`.
   The team size may be determined lazily at launch time, taking into account properties of the functor.
   Use the provided execution space instance during a kernel launch.  

### Runtime Settings

  * ```c++
    inline TeamPolicy& set_chunk_size(int chunk);
    ```
    Set the chunk size. Each physical team of threads will get assigned `chunk` consecutive teams. Default is 1.

  * ```c++
    inline TeamPolicy& set_scratch_size(const int& level, const Impl::PerTeamValue& per_team);
    inline TeamPolicy& set_scratch_size(const int& level, const Impl::PerThreadValue& per_thread);
    inline TeamPolicy& set_scratch_size(const int& level, const Impl::PerTeamValue& per_team, const Impl::PerThreadValue& per_thread);
    inline TeamPolicy& set_scratch_size(const int& level, const Impl::PerThreadValue& per_thread, const Impl::PerTeamValue& per_team);
    ```
    Set the per team and per thread scratch size. 
    * `level`: set the storage level. 0 is closest cache. 1 is closest storage (e.g. high bandwidth memory)
    * `per_team`: wrapper for the per team size of scratch in bytes. Returned by the function `PerTeam(int)`.
    * `per_thread`: wrapper for the per thread size of scratch in bytes. Returned by the function `PerThread(int)`.
    One can set the scratch size for level 0 and 1 independently by calling the function twice. 
    Subsequent calls with the same level overwrite the previous value. 

## Examples

  ```c++
    TeamPolicy<> policy_1(N,AUTO);
    TeamPolicy<Cuda> policy_2(N,T);
    TeamPolicy<Schedule<Dynamic>, OpenMP> policy_3(N,AUTO,8);
    TeamPolicy<IndexType<int>, Schedule<Dynamic>> policy_4(N,1,4);
    TeamPolicy<OpenMP> policy_5(OpenMP(), N, AUTO);
  ```